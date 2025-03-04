import asyncio
from PyPDF2 import PdfReader
import streamlit as st
from typing import List, Optional, Union, Literal
import nest_asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from bson import ObjectId
import requests
from io import BytesIO
import time

import logfire
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext

nest_asyncio.apply()

# Load environment variables
load_dotenv(override=True)

model = OpenAIModel(os.getenv("MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
logfire.configure(send_to_logfire='if-token-present')

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["candidate_interviews"]
candidates = db["candidates"]
job_descriptions = db["job_desc"]


class InterviewQuestionType(BaseModel):
    """Type of interview question format"""
    type: Literal["short_text", "multiple_choice", "yes_no"] = Field(
        description="The type of question format"
    )


class InterviewQuestion(BaseModel):
    """Details of an interview question to ask a candidate."""
    question: str = Field(description="The specific question to ask the candidate")
    question_type: InterviewQuestionType = Field(description="The format of the question")
    time: int = Field(description="The time allowed to answer the question in minutes (1-5)")
    options: Optional[List[str]] = Field(
        None, 
        description="Answer options for a multiple choice question, required only for multiple_choice type questions"
    )
    skill_assessed: str = Field(description="The specific job skill or requirement being assessed by this question")
    difficulty: Literal["basic", "intermediate", "advanced"] = Field(description="The difficulty level of the question")


class InterviewCriteria(BaseModel):
    """Evaluation criteria for the interview."""
    technical_skills: List[str] = Field(description="List of technical skills to evaluate from the job description")
    soft_skills: List[str] = Field(description="List of soft skills to evaluate from the job description")
    experience_requirements: List[str] = Field(description="Key experience requirements from the job description")
    scoring_matrix: str = Field(description="Detailed scoring matrix for evaluating answers")


class InterviewDetails(BaseModel):
    """Structure for an interview session with a candidate."""
    job_description: str
    candidate_information: str


class InterviewResponse(BaseModel):
    """Evaluation of a candidate's answer."""
    question: str
    answer: str
    evaluation: str
    technical_score: int = Field(description="The technical skill score (1-100)")
    relevance_score: int = Field(description="How relevant the answer was to the question (1-100)")
    completeness_score: int = Field(description="How complete the answer was (1-100)")
    overall_score: int = Field(description="The overall score (1-100) of the response")
    skill_feedback: str = Field(description="Specific feedback on the skill being assessed")


class InterviewSummary(BaseModel):
    """Summary evaluation of the entire interview."""
    technical_strengths: List[str] = Field(description="Technical strengths demonstrated by the candidate")
    technical_weaknesses: List[str] = Field(description="Technical areas for improvement")
    overall_fit_score: int = Field(description="Overall job fit score (1-100)")
    recommendation: str = Field(description="Hiring recommendation and justification")
    key_observations: List[str] = Field(description="Key observations from the interview")


# Function to extract text from a PDF (for the resume)
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Main interviewer agent with improved system prompt
interviewer_agent = Agent(
    model=model,
    deps_type=InterviewDetails,
    retries=5,
    system_prompt=(
        "You are a professional technical interviewer conducting a job-specific interview. "
        "Your task is to assess the candidate's qualifications for the specific role described in the job description. "
        "Begin by using the generate_evaluation_criteria tool to create a detailed evaluation framework based on the job requirements. "
        "This framework will guide your questioning and assessment throughout the interview. "
        "Ask questions that directly assess the required skills, knowledge, and experience from the job description. "
        "Use the candidate's resume information to tailor questions to their background and identify potential gaps. "
        "For each question: "
        "1. Use generate_question to create a role-specific question "
        "2. Wait for the candidate's response "
        "3. Use evaluate_response to assess their answer against the job requirements "
        "Ask 5-6 questions in total, mixing technical skills assessment with job-specific scenarios. "
        "Vary question formats (multiple choice, short text, yes/no) as appropriate for the skill being assessed. "
        "Complete the interview by using evaluate_interview to provide a comprehensive assessment of the candidate's fit for the role. "
        "Finally, use save_results to store the interview data and evaluation."
    ),
)


# Question generator with enhanced job specificity
question_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewQuestion,
    system_prompt=(
        "You are an expert technical interviewer who creates highly targeted questions to assess job candidates. "
        "Your task is to generate a specific job-relevant question based on the provided job description and candidate resume. "
        
        "First, analyze the job description to identify: "
        "1. Key technical skills required (programming languages, tools, frameworks, methodologies) "
        "2. Experience requirements (years, domains, project types) "
        "3. Core responsibilities of the position "
        "4. Problem-solving abilities needed "
        
        "Then, review the candidate's resume to understand: "
        "1. Their current technical skill set "
        "2. Experience level and background "
        "3. Potential knowledge gaps related to the job requirements "
        "4. Areas where deeper assessment is needed "
        
        "Create a question that directly tests a specific job requirement, choosing a format appropriate for the skill being tested: "
        "- Use multiple_choice for knowledge verification questions "
        "- Use short_text for conceptual understanding or problem-solving questions "
        "- Use yes_no for experience verification "
        
        "Make technical questions precise and focused on a single concept or skill. "
        "For multiple choice questions, provide 3-5 options with one clear correct answer. "
        "Ensure the question has an appropriate difficulty level and can be reasonably answered in the allocated time. "
        "Specify exactly which job skill or requirement is being assessed by this question. "
        "Vary the difficulty level across different questions to comprehensively assess the candidate."
    )
)


# Evaluation criteria generator with job-specific focus
criteria_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewCriteria,
    system_prompt=(
        "You are an expert in creating job-specific evaluation frameworks for technical interviews. "
        "Your task is to analyze the provided job description and candidate resume to create a detailed evaluation criteria matrix. "
        
        "First, thoroughly analyze the job description to extract: "
        "1. All required technical skills (languages, frameworks, tools, methodologies) "
        "2. Necessary soft skills (communication, teamwork, problem-solving) "
        "3. Experience requirements (years, industry knowledge, specific domains) "
        "4. Key responsibilities of the role "
        
        "Then create a comprehensive evaluation framework that includes: "
        "1. A detailed list of technical skills to be assessed, directly derived from the job requirements "
        "2. Soft skills relevant to the specific role "
        "3. Experience requirements that should be verified "
        "4. A detailed scoring matrix that defines what constitutes excellent, good, average, and poor responses "
        "   for this specific role "
        
        "The scoring matrix should be explicit about what demonstrates mastery in each required area. "
        "Focus on creating criteria that will objectively measure the candidate's fit for THIS SPECIFIC ROLE, "
        "not generic interview criteria. "
        "The criteria must be directly traceable to specific requirements in the job description."
    )
)


# Response evaluator with job-specific assessment
response_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewResponse,
    system_prompt=(
        "You are an expert technical evaluator assessing candidate responses against specific job requirements. "
        "Your task is to provide an objective, detailed evaluation of a candidate's answer based on: "
        "1. The job's specific technical requirements "
        "2. The skill being assessed by the question "
        "3. The evaluation criteria established for this role "
        
        "For each response, analyze: "
        "1. Technical accuracy: Is the answer technically correct based on industry standards? "
        "2. Relevance: Does the answer address the specific skill being assessed? "
        "3. Depth: Does the candidate demonstrate appropriate depth of knowledge for the role level? "
        "4. Practical application: Does the candidate show how they've applied this knowledge? "
        
        "Provide scores on a scale of 1-100 for technical skill, relevance, completeness, and overall quality. "
        "Give specific feedback on the skill being assessed, highlighting strengths and areas for improvement. "
        "Ground your evaluation in the specific job requirements, not general interview standards. "
        "Be fair but thorough in identifying gaps between the candidate's demonstrated knowledge and the job requirements."
    )
)


# Summative evaluator for overall assessment
summative_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewSummary,
    system_prompt=(
        "You are a senior technical hiring manager making a final assessment of a candidate's fit for a specific role. "
        "Review all the questions, answers, and individual evaluations from the interview to create a comprehensive assessment. "
        
        "Your evaluation should: "
        "1. Identify clear technical strengths where the candidate meets or exceeds job requirements "
        "2. Highlight technical areas where the candidate may need development relative to job requirements "
        "3. Assess overall fit for the specific role based on the job description requirements "
        "4. Provide a clear hiring recommendation with justification "
        "5. Note key observations about the candidate's technical abilities and potential "
        
        "Focus on how well the candidate's demonstrated skills and knowledge align with the SPECIFIC job requirements. "
        "Consider both the technical skills and the level of expertise required. "
        "Be objective and evidence-based, citing specific answers that support your assessment. "
        "The overall fit score should reflect how closely the candidate's demonstrated abilities match the key requirements. "
        "Provide actionable insights that would help the hiring team make an informed decision."
    )
)


@interviewer_agent.tool
async def generate_question(ctx: RunContext[InterviewDetails]) -> InterviewQuestion:
    """Generate a job-specific interview question based on the job description and resume."""
    r = await question_generator.run(
        "Generate a job-specific question that directly tests a requirement from the job description. "
        "Consider what has already been asked and test a different skill or requirement.",
        deps=ctx.deps
    )
    st.session_state.current_question_time = r.data.time
    st.session_state.current_question_type = r.data.question_type.type
    return r.data


@interviewer_agent.tool
async def generate_evaluation_criteria(ctx: RunContext[InterviewDetails]) -> InterviewCriteria:
    """Create a detailed evaluation framework based on the specific job requirements."""
    r = await criteria_generator.run(
        "Create a comprehensive evaluation criteria matrix specifically tailored to this job role. "
        "Extract the key technical skills, experience requirements, and responsibilities from the job description.",
        deps=ctx.deps
    )
    st.session_state.evaluation_criteria = r.data
    return r.data


@interviewer_agent.tool(docstring_format='google')
async def evaluate_response(ctx: RunContext[InterviewDetails], criteria, question, answer, skill_assessed) -> InterviewResponse:
    """Evaluate the candidate's answer against the job-specific criteria.

    Args:
        criteria: The evaluation criteria for this role
        question: The specific question asked
        answer: The candidate's response
        skill_assessed: The job skill being assessed by this question
    """
    r = await response_evaluator.run(
        f"""Evaluate the candidate's answer based on the job requirements:
        
        Job Requirements: The relevant section from the job description that relates to this skill is:
        {ctx.deps.job_description}
        
        Question: {question}
        Skill Being Assessed: {skill_assessed}
        Answer: {answer}
        
        Evaluation Criteria: {criteria}
        
        Provide a detailed, job-specific evaluation focusing on how well the answer demonstrates the required skill.""",
        deps=ctx.deps
    )
    st.session_state.responses.append(r.data.model_dump())
    return r.data


@interviewer_agent.tool
async def evaluate_interview(ctx: RunContext[InterviewDetails]) -> InterviewSummary:
    """Provide a comprehensive assessment of the candidate's fit for the specific role."""
    r = await summative_evaluator.run(
        f"""Create a comprehensive evaluation of the candidate's fit for this specific role based on:
        
        Job Description: {ctx.deps.job_description}
        
        Candidate Resume: {ctx.deps.candidate_information}
        
        Interview Responses: {st.session_state.responses}
        
        Provide a detailed assessment focusing on the match between the candidate's demonstrated abilities and the specific job requirements.""",
        deps=ctx.deps
    )
    return r.data


@interviewer_agent.tool(docstring_format='google')
async def save_results(ctx: RunContext[InterviewDetails], summary: InterviewSummary) -> str:
    """Save the interview results and final evaluation.

    Args:
        summary: The comprehensive interview evaluation and hiring recommendation
    """
    if st.query_params:
        interview_data = {
            "candidate_id": st.query_params['candidate_id'],
            "job_desc_id": st.query_params['job_desc_id'],
            "responses": st.session_state.responses,
            "summary": summary.model_dump()
        }
    else:
        interview_data = {
            "resume": st.session_state.interview_details.candidate_information,
            "job_description": st.session_state.interview_details.job_description,
            "responses": st.session_state.responses,
            "summary": summary.model_dump()
        }
    collection.insert_one(interview_data)
    return "Interview results saved successfully"


@interviewer_agent.system_prompt
async def get_interview_detail(ctx: RunContext[InterviewDetails]) -> str:
    return f"""
    JOB DESCRIPTION:
    {ctx.deps.job_description}
    
    CANDIDATE RESUME:
    {ctx.deps.candidate_information}
    
    Your task is to conduct a targeted technical interview for this specific role. Focus on assessing the candidate's qualifications against the job requirements.
    """


async def display_timer(timer, seconds_remaining):
    while True:
        # Timer
        time_remaining = seconds_remaining - (time.time() - st.session_state.start_time)
        if time_remaining > 0:
            timer.markdown(f"Time remaining: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Time's up. Moving to the next question.")
            with st.spinner("Processing your response..."):
                result = await interviewer_agent.run(
                    user_prompt="Time's up, please evaluate this answer and continue to the next question.",
                    deps=st.session_state.interview_details,
                    message_history=st.session_state.history
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data
                st.session_state.start_time = time.time()
                st.rerun()


async def main():
    st.title("AI Technical Interview System")
    
    # UI improvements
    st.markdown("""
    This system conducts a job-specific technical interview based on the job description and your resume.
    The questions will be tailored to assess your fit for the specific role.
    """)

    job_description = None
    resume_file = None

    if st.query_params:
        try:
            job_desc = job_descriptions.find_one(ObjectId(st.query_params['job_desc_id']))
            job_description = job_desc['job_desc']

            # Retrieve the file URL from MongoDB
            candidate = candidates.find_one(ObjectId(st.query_params['candidate_id']))

            if candidate:
                file_url = candidate.get("resume")
                response = requests.get(file_url)

                if response.status_code == 200:
                    resume_file = BytesIO(response.content)
                else:
                    st.error(f"Failed to download resume. Status code: {response.status_code}")
            else:
                st.error("Candidate information not found")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if not st.query_params and not (job_description and resume_file):
        # Upload job description
        job_description = st.text_area("Enter the job description:", height=200, 
                                     help="Paste the complete job description including requirements and responsibilities")

        # Upload resume
        resume_file = st.file_uploader("Upload your resume (PDF):", type="pdf",
                                     help="Upload your current resume in PDF format")

    if job_description and resume_file:
        # Extract text from the resume
        resume_text = extract_text_from_pdf(resume_file)

        # Initialize session state variables
        if "interview_details" not in st.session_state:
            interview_details = InterviewDetails(
                job_description=job_description,
                candidate_information=resume_text
            )
            st.session_state.interview_details = interview_details
        
        if "history" not in st.session_state:
            st.session_state.history = []
        
        if "display_msg" not in st.session_state:
            st.session_state.display_msg = ""
        
        if "responses" not in st.session_state:
            st.session_state.responses = []
        
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()
        
        if "current_question_time" not in st.session_state:
            st.session_state.current_question_time = 2
            
        if "current_question_type" not in st.session_state:
            st.session_state.current_question_type = "short_text"
            
        if "evaluation_criteria" not in st.session_state:
            st.session_state.evaluation_criteria = None

        # Start the interview process
        if len(st.session_state.history) == 0:
            if st.button("Start Interview", type="primary"):
                with st.spinner("Preparing interview questions based on the job description..."):
                    result = await interviewer_agent.run(
                        user_prompt="Hello, I'm ready to start the interview for this specific role.",
                        deps=st.session_state.interview_details
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    st.rerun()
        else:
            # Display current question and timer
            timer_container = st.empty()
            
            with st.chat_message("interviewer", avatar="👩‍💼"):
                st.write(st.session_state.display_msg)
            
            # Handle different question types
            if st.session_state.current_question_type == "multiple_choice":
                # Extract options from the message
                import re
                message = st.session_state.display_msg
                options = []
                
                # Simple pattern matching to extract options
                option_pattern = r'[A-Z]\)\s+(.*?)(?=\s+[A-Z]\)|$)'
                matches = re.findall(option_pattern, message)
                if matches:
                    options = [option.strip() for option in matches]
                
                if not options:
                    # Fallback to numbered options if lettered options aren't found
                    option_pattern = r'\d+\.\s+(.*?)(?=\s+\d+\.|$)'
                    matches = re.findall(option_pattern, message)
                    if matches:
                        options = [option.strip() for option in matches]
                
                # If options were successfully extracted
                if options:
                    answer = st.radio("Select your answer:", options)
                    if st.button("Submit Answer"):
                        with st.spinner("Evaluating your response..."):
                            result = await interviewer_agent.run(
                                user_prompt=f"My answer is: {answer}",
                                deps=st.session_state.interview_details,
                                message_history=st.session_state.history
                            )
                            st.session_state.history = result.all_messages()
                            st.session_state.display_msg = result.data
                            st.session_state.start_time = time.time()
                            st.rerun()
                else:
                    # If option extraction failed, fall back to text input
                    answer = st.text_input("Your answer:")
                    if st.button("Submit Answer"):
                        with st.spinner("Evaluating your response..."):
                            result = await interviewer_agent.run(
                                user_prompt=answer,
                                deps=st.session_state.interview_details,
                                message_history=st.session_state.history
                            )
                            st.session_state.history = result.all_messages()
                            st.session_state.display_msg = result.data
                            st.session_state.start_time = time.time()
                            st.rerun()
            
            elif st.session_state.current_question_type == "yes_no":
                answer = st.radio("Your answer:", ["Yes", "No"])
                if st.button("Submit Answer"):
                    with st.spinner("Evaluating your response..."):
                        result = await interviewer_agent.run(
                            user_prompt=f"My answer is: {answer}",
                            deps=st.session_state.interview_details,
                            message_history=st.session_state.history
                        )
                        st.session_state.history = result.all_messages()
                        st.session_state.display_msg = result.data
                        st.session_state.start_time = time.time()
                        st.rerun()
            
            else:  # short_text or default
                answer = st.text_area("Your answer:", height=100)
                if st.button("Submit Answer"):
                    with st.spinner("Evaluating your response..."):
                        result = await interviewer_agent.run(
                            user_prompt=answer,
                            deps=st.session_state.interview_details,
                            message_history=st.session_state.history
                        )
                        st.session_state.history = result.all_messages()
                        st.session_state.display_msg = result.data
                        st.session_state.start_time = time.time()
                        st.rerun()
            
            # Display timer
            asyncio.run(display_timer(timer_container, 60 * st.session_state.current_question_time))

if __name__ == "__main__":
    asyncio.run(main())
