import asyncio
from PyPDF2 import PdfReader
import streamlit as st
from typing import List, Optional, Dict, Literal, Union
import nest_asyncio
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import os
from bson import ObjectId
import requests
from io import BytesIO
import time
import copy

import logfire
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent, RunContext

nest_asyncio.apply()

# Load environment variables
load_dotenv(override=True)

model = OpenAIModel(os.getenv("MODEL"), api_key=os.getenv("O3_API_KEY"))
logfire.configure(send_to_logfire='if-token-present')

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["candidate_interviews"]
candidates = db["candidates"]
job_descriptions = db["job_desc"]


class QuestionType(BaseModel):
    """Type of interview question."""
    type: Literal["short_answer", "multiple_choice", "yes_no"]


class InterviewQuestion(BaseModel):
    """Details of an interview question to ask a candidate."""
    question: str = Field(description="The specific question to ask the candidate")
    question_type: Literal["short_answer", "multiple_choice", "yes_no"] = Field(
        description="The type of question format"
    )
    time: Optional[int] = Field(description="The time required to answer the question in minutes", default=3)
    options: Optional[List[str]] = Field(
        description="Answer options for a multiple choice question",
        default=None
    )
    skill_assessed: Optional[str] = Field(
        description="The specific skill or qualification from the job description being assessed"
    )
    difficulty_level: Optional[Literal["basic", "intermediate", "advanced"]] = Field(
        description="The difficulty level of the question"
    )


class JobRequirement(BaseModel):
    """Structure for job requirements extracted from a job description."""
    technical_skills: List[str] = Field(description="Technical skills required for the job")
    soft_skills: List[str] = Field(description="Soft skills required for the job")
    experience: List[str] = Field(description="Required experience levels")
    responsibilities: List[str] = Field(description="Key job responsibilities")
    qualifications: List[str] = Field(description="Educational qualifications and certifications")


class CandidateProfile(BaseModel):
    """Structure for candidate information extracted from a resume."""
    skills: List[str] = Field(description="Skills mentioned in the resume")
    experience: List[str] = Field(description="Work experience highlighted in the resume")
    education: List[str] = Field(description="Educational background from the resume")
    achievements: List[str] = Field(description="Notable achievements from the resume")


class InterviewDetails(BaseModel):
    """Structure for an interview session with a candidate."""
    job_description: str = Field(description="The full text of the job description")
    candidate_information: str = Field(description="The full text of the candidate's resume")
    # job_requirements: Optional[JobRequirement] = Field(description="Structured job requirements", default=None)
    # candidate_profile: Optional[CandidateProfile] = Field(description="Structured candidate profile", default=None)


class EvaluationCriteria(BaseModel):
    """Structure for evaluation criteria based on job requirements."""
    technical_assessment: Dict[str, str] = Field(
        description="Technical assessment component containing criteria for evaluating hard skills specific to the role, such as programming languages, tools, platforms, and methodologies with skill name as key and criteria as value. Dictionary mapping specific technical skills (e.g., 'Python', 'AWS', 'React') to detailed evaluation criteria for each skill. The criteria should define the expected proficiency level and specific capabilities."
    )
    experience_assessment: Dict[str, str] = Field(
        description="Experience assessment component for evaluating relevant work history, including industry-specific experience, role-related responsibilities, and achievements in similar positions. Dictionary mapping experience categories (e.g., 'Industry Experience', 'Leadership', 'Project Management') to specific requirements and expectations for each category. Should include both quantitative (years) and qualitative aspects."
    )
    communication_assessment: str = Field(
        description="Detailed criteria for evaluating verbal and written communication abilities, including clarity, conciseness, audience adaptation, technical communication, and presentation skills."
    )
    problem_solving_assessment: str = Field(
        description="Comprehensive criteria for assessing analytical thinking, creative problem-solving approach, solution evaluation methods, and ability to handle ambiguity and constraints."
    )
    job_fit_assessment: str = Field(
        description="Holistic criteria for evaluating candidate alignment with company culture, team dynamics, growth potential within the organization, and long-term career objectives relative to the position."
    )
    scoring_rubric: Dict[str, str] = Field(
        description="Comprehensive scoring system that defines how each assessment area should be evaluated numerically, allowing for standardized comparison between candidates across all evaluation dimensions. Dictionary mapping assessment categories (e.g., 'technical', 'experience', 'soft_skills') to their respective ScoringLevel objects. This provides a structured way to score each major evaluation area."
    )


class InterviewResponse(BaseModel):
    """Structure for evaluating candidate responses."""
    question: str = Field(description="The question that was asked")
    question_type: str = Field(description="The type of question that was asked")
    skill_assessed: str = Field(description="The skill being assessed by this question")
    answer: str = Field(description="The candidate's answer")
    evaluation: str = Field(description="Detailed evaluation of the answer")
    strengths: Optional[List[str]] = Field(description="Strengths identified in the answer")
    areas_for_improvement: Optional[List[str]] = Field(description="Areas where the answer could be improved")
    score: int = Field(description="The score of the response ranging 1-100")


class InterviewSummary(BaseModel):
    """Structure for the overall interview assessment."""
    technical_skills_assessment: List[str] = Field(
        description="List of specific technical skills (e.g., 'Python', 'SQL', 'System Design') to their numerical scores (typically 1-100, where 100 is highest proficiency). Only include skills that were assessed during the interview."
    )
    soft_skills_assessment: List[str] = Field(
        description="List of soft skills (e.g., 'Communication', 'Teamwork', 'Problem-solving') to their numerical scores (typically 1-100, where 100 is highest proficiency). Only include skills that were observed during the interview."
    )
    overall_score: int = Field(
        description="Aggregate numerical score representing the candidate's overall interview performance on a scale from 1 to 100, where higher scores indicate stronger performance.",
        ge=1,
        le=100
    )
    strengths: Optional[List[str]] = Field(
        description="List of concise, specific phrases identifying the candidate's most notable strengths demonstrated during the interview. Each strength should be a single clear statement without explanations."
    )
    areas_for_improvement: Optional[List[str]] = Field(
        description="List of concise, specific phrases identifying areas where the candidate showed weakness or could improve. Each area should be a single clear statement without explanations."
    )
    job_fit_assessment: str = Field(
        description="Concise paragraph (1-3 sentences) evaluating how well the candidate's skills, experience, and attributes align with the specific job requirements and team dynamics."
    )
    hiring_recommendation: Literal["Strong Yes", "Yes", "Maybe", "No", "Strong No"] = Field(
        description="Single categorical assessment of whether to hire the candidate, selected from exactly one of these five options: 'Strong Yes', 'Yes', 'Maybe', 'No', or 'Strong No'."
    )


# Function to extract text from a PDF (for the resume)
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Agent for parsing structured information from job description and resume
parser_agent = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=Union[JobRequirement, CandidateProfile],
    system_prompt=(
        "You are an expert at parsing job descriptions and resumes into structured information. "
        "When given a job description, extract key requirements, skills, and responsibilities. "
        "When given a resume, extract skills, experience, education, and achievements."
    )
)

# Improved interviewer agent
interviewer_agent = Agent(
    model=model,
    deps_type=InterviewDetails,
    retries=2,
    system_prompt=(
        "You are an expert technical interviewer who specializes in conducting role-specific interviews. "
        "Your goal is to assess candidates based on the specific requirements in the job description. "
        "You MUST follow these steps in order:\n"
        "1. First, use the parse_inputs tool to extract structured information from the job description and resume.\n"
        "2. Next, use the generate_evaluation_criteria tool to create a detailed, role-specific evaluation matrix.\n"
        "3. Next, use the generate_question tool to generate a targeted, job-specific question and ask it from the candidate.\n"
        "4. Evaluate each response with the evaluate_response tool.\n"
        "5. Provide a comprehensive assessment at the end using the evaluate_interview tool.\n\n"
        "Guidelines:\n"
        "- Ask questions in Mongolian language, the candidate will also respond in Mongolian.\n"
        "- Ask only one question at a time and wait for the answer.\n"
        "- Focus on technical skills specific to the job description.\n"
        "- Use different question types (short answer, multiple choice, yes/no)\n"
        "- Ask questions of varying difficulty levels\n"
        "- Do not ask the same question twice\n"
        "- Do not ask the same type of question twice\n"
        "- Do NOT pass a question to the generate_question tool.\n"
        "- You must avoid generic behavioral questions and behavioral questions altogether\n"
        "- Conclude the interview after 6 questions\n"
        "- Save all results using the save_results tool"
    ),
)

# Improved question generator
question_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewQuestion,
    system_prompt=(
        "You are an expert at creating job-specific interview questions. Generate a question in Mongolian language that:"
        "\n1. Directly relates to a SPECIFIC technical skill, qualification, or responsibility from the job description"
        "\n2. Is appropriate for the candidate's background based on their resume"
        "\n3. Has the right difficulty level and only in [short answer, multiple choice, or yes/no] types"
        "\n4. Provides options if it's a multiple-choice question"
        "\n5. Assesses the candidate's actual abilities, not just knowledge"
        "\n6. Is concise and clear, as a real interviewer would ask"
        "\n7. Avoids asking the same question as previous ones"
        "\n8. Identifies which specific skill or requirement is being assessed"
        "\n9. Sets an appropriate time limit for answering"
        "\n\nDO NOT create generic questions that could be asked in any interview."
        "\nLook at the structured job requirements and candidate profile to make your question relevant."
    )
)

# Improved criteria generator
criteria_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=EvaluationCriteria,
    retries=5,
    system_prompt=(
        "You are an expert at creating evaluation criteria for technical interviews. "
        "Create a detailed, role-specific evaluation matrix that:"
        "\n1. Includes criteria for SPECIFIC technical skills listed in the job description relevant to the role"
        "\n2. Includes criteria for assessing experience requirements"
        "\n3. Includes criteria for assessing problem-solving abilities relevant to the role"
        "\n4. Includes criteria for assessing communication in a role-specific context"
        "\n5. Provides a detailed scoring rubric for each assessment area"
        "\n6. Is specifically tailored to the job requirements and responsibilities"
        "\n\nUse the structured job requirements to make your criteria concrete and specific."
        "\nAvoid generic evaluation criteria that could apply to any job."
        "\nReturn a JSON in the following format:"
        '\n{"job_fit_assessment": <str>, "experience_assessment": {<job_requirement>: <str>,...}, "technical_assessment": {<job_requirement>: <str>,...}, "scoring_rubric": {<score>: <explanation>,...}}'
    )
)

# Improved response evaluator
response_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewResponse,
    system_prompt=(
        "You are an expert at evaluating candidate responses in technical interviews. "
        "Your evaluation should:"
        "\n1. Compare the answer against the specific job requirements and skills needed"
        "\n2. Assess the technical accuracy and depth of understanding shown"
        "\n3. Identify specific strengths in the response"
        "\n4. Identify specific areas for improvement"
        "\n5. Provide a fair score based on the established evaluation criteria"
        "\n6. Consider the candidate's background from their resume"
        "\n7. Be objective and constructive"
        "\n\nEnsure your evaluation directly ties back to the specific skill being assessed and its importance for the role."
        "\nYou must respond in Mongolian language."

    )
)

# Improved summary evaluator
summary_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewSummary,
    retries=5,
    system_prompt=(
        "You are an expert at providing comprehensive interview assessments. "
        "Create a detailed summary that:"
        "\n1. Evaluates the candidate's demonstrated technical skills"
        "\n2. Assesses soft skills relevant to the role"
        "\n3. Calculates an overall score based on all responses"
        "\n4. Identifies the candidate's key strengths related to job requirements"
        # "\n5. Identifies specific areas for improvement"
        "\n5. Provides a clear assessment of job fit"
        "\n6. Makes a hiring recommendation based on all the evidence"
        "\n\nYour assessment should be directly tied to the specific job requirements and responsibilities."
        "\nAvoid generic assessments that could apply to any candidate or role."
        "\nYou must respond in Mongolian language."
    )
)


@interviewer_agent.tool
async def parse_inputs(ctx: RunContext[InterviewDetails]) -> str:
    """Parse the job description and resume to extract structured information."""
    # Parse job description
    job_result = await parser_agent.run(
        "Parse the following job description into structured requirements:\n" + ctx.deps.job_description,
        deps=ctx.deps
    )
    st.session_state.job_requirements = job_result.data

    # Parse resume
    resume_result = await parser_agent.run(
        "Parse the following resume into a structured candidate profile:\n" + ctx.deps.candidate_information,
        deps=ctx.deps
    )
    st.session_state.candidate_profile = resume_result.data

    return "Inputs parsed successfully. Job requirements and candidate profile extracted."


@interviewer_agent.tool(require_parameter_descriptions=False)
async def generate_question(ctx: RunContext[InterviewDetails]) -> InterviewQuestion:
    """Generate a job-specific interview question."""
    prompt = """
    Generate a job-specific interview question based on the job requirements and candidate profile.

    Job requirements: {job_requirements}

    Candidate profile: {candidate_profile}

    Previous questions: {previous_questions}

    Create a question that directly assesses a specific skill or qualification needed for this role.
    Do not generate a same type of question in a row.
    """

    previous_questions = [r.get("question", "") for r in
                          st.session_state.responses] if "responses" in st.session_state else []

    r = await question_generator.run(
        prompt.format(
            job_requirements=st.session_state.job_requirements,
            candidate_profile=st.session_state.candidate_profile,
            previous_questions=previous_questions
        ),
        deps=ctx.deps
    )
    st.session_state.current_question_time = r.data.time
    return r.data


@interviewer_agent.tool
async def generate_evaluation_criteria(ctx: RunContext[InterviewDetails]) -> EvaluationCriteria:
    """Generate detailed evaluation criteria based on job requirements."""
    prompt = """
        Create detailed evaluation criteria specific to this role and its requirements.

        Job requirements: {job_requirements}
        """
    while not 'job_requirements' in st.session_state:
        await asyncio.sleep(1)
    r = await criteria_generator.run(
        prompt.format(
            job_requirements=st.session_state.job_requirements
        ),
        deps=ctx.deps
    )

    st.session_state.evaluation_criteria = r.data
    return r.data


@interviewer_agent.tool
async def evaluate_response(
        ctx: RunContext[InterviewDetails],
        question: str,
        question_type: str,
        skill_assessed: str,
        answer: str
) -> InterviewResponse:
    """Evaluate the candidate's answer based on the job-specific criteria."""
    prompt = """
    Evaluate the candidate's response based on the established criteria and job requirements.

    Question: {question}
    Question Type: {question_type}
    Skill Being Assessed: {skill_assessed}
    Answer: {answer}

    Evaluation Criteria: {criteria}

    Job Requirements: {job_requirements}

    Provide a detailed, job-specific evaluation.
    """

    r = await response_evaluator.run(
        prompt.format(
            question=question,
            question_type=question_type,
            skill_assessed=skill_assessed,
            answer=answer,
            criteria=st.session_state.evaluation_criteria if "evaluation_criteria" in st.session_state else "Not available",
            job_requirements=st.session_state.job_requirements
        ),
        deps=ctx.deps
    )
    st.session_state.responses.append(r.data.model_dump())
    return r.data


@interviewer_agent.tool
async def evaluate_interview(ctx: RunContext[InterviewDetails]) -> InterviewSummary:
    """Provide a comprehensive assessment of the entire interview."""
    prompt = """
    Create a comprehensive summary evaluation of the entire interview.

    Job Requirements: {job_requirements}
    Candidate Profile: {candidate_profile}
    All Responses: {responses}

    Provide a detailed assessment of how well the candidate meets each of the specific requirements of this role.
    Format your output as a matrix with evaluated job requirement, question asked, response, score, and evaluation as columns.

    """

    r = await summary_evaluator.run(
        prompt.format(
            job_requirements=st.session_state.job_requirements,
            candidate_profile=st.session_state.candidate_profile,
            responses=st.session_state.responses if "responses" in st.session_state else []
        ),
        deps=ctx.deps
    )
    return r.data


@interviewer_agent.tool
async def save_results(ctx: RunContext[InterviewDetails], summary: InterviewSummary) -> str:
    """Save the interview results and final assessment."""
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
    return "Interview results saved successfully."


@interviewer_agent.system_prompt
async def get_interview_detail(ctx: RunContext[InterviewDetails]) -> str:
    return f"Interview details: {ctx.deps.model_dump()}"


async def print_timer(job_description, timer, s):
    while True:
        time_remaining = s - (time.time() - st.session_state.start_time)
        if time_remaining > 0:
            timer.markdown(f"Time remaining: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Time's up. Moving to the next question.")
            with st.spinner("Please wait..."):
                result = await interviewer_agent.run(
                    user_prompt="No answer.",
                    deps=st.session_state.interview_details,
                    message_history=st.session_state.history
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data
                st.session_state.start_time = time.time()
                st.rerun()


Agent.instrument_all()


async def main():
    st.title("Job-Specific AI Interview System")

    job_description = None
    resume_file = None

    if st.query_params:
        try:
            job_desc = job_descriptions.find_one(ObjectId(st.query_params['job_desc_id']))
            job_description = job_desc['job_desc']

            candidate = candidates.find_one(ObjectId(st.query_params['candidate_id']))
            if candidate:
                file_url = candidate.get("resume")
                response = requests.get(file_url)
                if response.status_code == 200:
                    resume_file = BytesIO(response.content)
                else:
                    st.error(f"Failed to download resume. Status code: {response.status_code}")
            else:
                st.error("Candidate record not found")
        except Exception as e:
            st.error(f"Error loading data: {e}")

    if not st.query_params and not (job_description and resume_file):
        st.info("Please provide job description and resume to begin the interview")
        job_description = st.text_area("Enter job description:", height=200)
        resume_file = st.file_uploader("Upload your resume (PDF):", type="pdf")

    if job_description and resume_file:
        resume_text = extract_text_from_pdf(resume_file)

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
            st.session_state.current_question_time = 3
        if "evaluation_criteria" not in st.session_state:
            st.session_state.evaluation_criteria = None

        if len(st.session_state.history) == 0:
            with st.spinner("Preparing interview questions based on job requirements..."):
                result = await interviewer_agent.run(
                    user_prompt="Hello, I'm ready for the interview.",
                    deps=st.session_state.interview_details
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data

            if st.button("Start Interview"):
                st.session_state.start_time = time.time()
                st.rerun()
        else:
            # Display question and handle responses
            timer = st.empty()

            with st.chat_message("assistant", avatar="👨‍💼"):
                st.write(st.session_state.display_msg)
                st.session_state.start_time = time.time()
                # Check if current message contains multiple choice options
                if "options:" in st.session_state.display_msg.lower() or "option" in st.session_state.display_msg.lower():
                    # Try to extract options
                    try:
                        options_text = st.session_state.display_msg.split("Options:", 1)[1].strip()
                        options = [opt.strip() for opt in options_text.split("\n") if opt.strip()]
                        # If we found options, display them as radio buttons
                        if options:
                            selected_option = st.radio("Select your answer:", options, key="mc_response")
                            if st.button("Submit Answer"):
                                with st.spinner("Evaluating your response..."):
                                    result = await interviewer_agent.run(
                                        user_prompt=f"My answer is: {selected_option}",
                                        deps=st.session_state.interview_details,
                                        message_history=st.session_state.history
                                    )
                                    st.session_state.history = result.all_messages()
                                    st.session_state.display_msg = result.data
                                    st.session_state.start_time = time.time()
                                    st.rerun()
                    except:
                        # If extraction fails, fall back to text input
                        pass

            # Only show text input if not showing multiple choice
            if "selected_option" not in locals():
                answer = st.chat_input("Your answer:")
                if answer:
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

            # Show timer
            asyncio.run(print_timer(job_description, timer, 60 * st.session_state.current_question_time))


if __name__ == "__main__":
    asyncio.run(main())
