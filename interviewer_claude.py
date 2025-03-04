import asyncio
import os
import time
from io import BytesIO
from typing import Dict, List, Optional

import logfire
import nest_asyncio
import requests
import streamlit as st
from bson import ObjectId
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from PyPDF2 import PdfReader
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from pymongo.server_api import ServerApi

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Load environment variables
load_dotenv(override=True)

# Initialize OpenAI model and logging
model = OpenAIModel(os.getenv("MODEL", "gpt-4"), api_key=os.getenv("OPENAI_API_KEY"))
logfire.configure(send_to_logfire='if-token-present')

# Application constants
INTERVIEW_LENGTH_MINUTES = 10
DEFAULT_QUESTION_TIME_MINUTES = 2
MAX_QUESTIONS = 6
DB_NAME = "interview_db"


# --- Data Models ---

class InterviewQuestion(BaseModel):
    """Details of an interview question to ask a candidate."""
    question: str
    time: int = Field(description="The time required to answer the question in minutes")
    options: Optional[List[str]] = Field(default=None, description="Answer options for a multiple choice question")


class InterviewDetails(BaseModel):
    """Structure for an interview session with a candidate."""
    job_description: str
    candidate_information: str


class InterviewResponse(BaseModel):
    """Evaluation of a candidate's response to a question."""
    question: str
    answer: str
    evaluation: str
    score: int = Field(description="The score (1-100) of the response.")


# --- Database Connection ---

class Database:
    """Database connection and operations manager."""
    
    def __init__(self):
        """Initialize database connection."""
        self.client = None
        self.db = None
        self.connect()
        
    def connect(self):
        """Establish connection to MongoDB."""
        try:
            mongo_uri = os.getenv("MONGO_URI")
            if not mongo_uri:
                st.error("MongoDB URI not found in environment variables.")
                return
                
            self.client = MongoClient(mongo_uri, server_api=ServerApi('1'))
            # Ping the server to confirm connection
            self.client.admin.command('ping')
            self.db = self.client[DB_NAME]
            st.session_state.db_connected = True
        except ConnectionFailure:
            st.error("Failed to connect to MongoDB. Please check your connection and try again.")
            st.session_state.db_connected = False
        except Exception as e:
            st.error(f"Database error: {str(e)}")
            st.session_state.db_connected = False
    
    def save_interview(self, interview_data: Dict) -> Optional[str]:
        """Save interview data to the database.
        
        Args:
            interview_data: Dictionary containing interview details and results
            
        Returns:
            String ID of the saved document or None if operation failed
        """
        if not self.db:
            st.warning("Database connection not available")
            return None
            
        try:
            result = self.db["candidate_interviews"].insert_one(interview_data)
            return str(result.inserted_id)
        except OperationFailure as e:
            st.error(f"Failed to save interview: {str(e)}")
            return None
    
    def get_candidate(self, candidate_id: str) -> Optional[Dict]:
        """Retrieve candidate information by ID.
        
        Args:
            candidate_id: String ID of the candidate
            
        Returns:
            Dictionary with candidate data or None if not found
        """
        if not self.db:
            return None
            
        try:
            return self.db["candidates"].find_one({"_id": ObjectId(candidate_id)})
        except Exception as e:
            st.error(f"Failed to retrieve candidate: {str(e)}")
            return None
    
    def get_job_description(self, job_id: str) -> Optional[Dict]:
        """Retrieve job description by ID.
        
        Args:
            job_id: String ID of the job description
            
        Returns:
            Dictionary with job description or None if not found
        """
        if not self.db:
            return None
            
        try:
            return self.db["job_desc"].find_one({"_id": ObjectId(job_id)})
        except Exception as e:
            st.error(f"Failed to retrieve job description: {str(e)}")
            return None


# --- File Handling ---

def extract_text_from_pdf(file) -> str:
    """Extract all text from a PDF file.
    
    Args:
        file: A file-like object containing PDF data
        
    Returns:
        String containing all extracted text
    """
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"Error extracting text from PDF: {str(e)}")
        return "Failed to extract text from resume."


def download_file(url: str) -> Optional[BytesIO]:
    """Download a file from a URL and return it as BytesIO.
    
    Args:
        url: The URL of the file to download
        
    Returns:
        BytesIO object containing the file or None if download failed
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to download file: {str(e)}")
        return None


# --- Agent Definitions ---

# Define the interviewer agent with improved prompts
interviewer_agent = Agent(
    model=model,
    deps_type=InterviewDetails,
    retries=3,
    system_prompt=(
        "You are a professional technical interviewer who specializes in assessing candidates "
        "for software development roles. Follow these guidelines strictly:\n\n"
        "1. Begin by generating evaluation criteria using the generate_evaluation_criteria tool\n"
        "2. Ask focused, technical questions about the candidate's skills relevant to the job description\n"
        "3. Do not repeat questions or ask generic behavioral questions\n"
        "4. Evaluate each response thoroughly using the evaluate_response tool\n"
        "5. Focus on depth of technical knowledge, problem-solving approach, and critical thinking\n"
        "6. Maintain a professional but conversational tone\n"
        "7. End the interview after asking 6 questions or when you reach the 10-minute mark\n"
        "8. Provide a comprehensive final evaluation using the evaluate_interview tool\n"
        "9. Save all results using the save_results tool\n\n"
        "Remember to be fair and objective in your assessment."
    ),
)

# Define the question generator agent with improved prompts
question_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewQuestion,
    system_prompt=(
        "You are an expert at creating technical interview questions. Your task is to generate "
        "challenging but fair questions based on the job description and candidate's resume.\n\n"
        "Guidelines for creating questions:\n"
        "1. Focus on technical skills directly relevant to the job description\n"
        "2. Vary question types (coding problems, system design, debugging scenarios, etc.)\n"
        "3. Adjust difficulty based on the candidate's experience level\n"
        "4. Make questions specific and actionable\n"
        "5. Avoid generic or overly theoretical questions\n"
        "6. For multiple choice questions, include 3-4 plausible options\n"
        "7. Assign an appropriate time limit between 1-3 minutes\n\n"
        "Each question should reveal the depth of the candidate's knowledge in a specific area."
    )
)

# Define the evaluation criteria generator
critic_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    system_prompt=(
        "You are an expert in technical interview evaluation methodologies. Create a detailed "
        "evaluation matrix specific to the job description and the candidate's resume.\n\n"
        "Your evaluation matrix should:\n"
        "1. Include 4-6 key dimensions relevant to the role (e.g., technical knowledge, problem-solving, code quality)\n"
        "2. Define clear scoring criteria for each dimension (1-100 scale)\n"
        "3. Specify what constitutes excellent, good, satisfactory, and poor performance\n"
        "4. Weight dimensions according to their importance for the specific role\n"
        "5. Include both technical skills and soft skills relevant to the position\n\n"
        "The matrix should be concise but comprehensive enough to provide fair and consistent evaluation."
    )
)

# Define the response evaluator
response_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewResponse,
    system_prompt=(
        "You are an expert technical evaluator assessing interview responses. Your evaluation should be:\n\n"
        "1. Objective and based on the pre-established criteria\n"
        "2. Specific in identifying strengths and weaknesses\n"
        "3. Balanced in considering both technical accuracy and communication quality\n"
        "4. Numerical (assign a score from 1-100) with clear justification\n"
        "5. Focused on evidence from the response rather than assumptions\n\n"
        "Provide detailed, actionable feedback that explains the score and highlights key observations."
    )
)

# Define the summative evaluator
summative_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    system_prompt=(
        "You are conducting a holistic evaluation of an entire technical interview. Your summative assessment should:\n\n"
        "1. Synthesize patterns across all responses\n"
        "2. Identify the candidate's key strengths and improvement areas\n"
        "3. Evaluate alignment with the specific job requirements\n"
        "4. Consider technical skills, problem-solving approach, and communication ability\n"
        "5. Provide a hiring recommendation with clear justification\n"
        "6. Suggest potential role fits if the candidate is strong but not ideal for the original position\n\n"
        "Be balanced, fair, and focus on evidence from the interview responses."
    )
)


# --- Agent Tools ---

@interviewer_agent.tool
async def generate_question(ctx: RunContext[InterviewDetails]) -> InterviewQuestion:
    """Generate a new interview question based on job requirements and candidate profile."""
    try:
        r = await question_generator.run(
            "Generate a technical interview question appropriate for this candidate and job role",
            deps=ctx.deps
        )
        # Set a reasonable default if no time is specified
        question_time = r.data.time if r.data.time > 0 else DEFAULT_QUESTION_TIME_MINUTES
        st.session_state.current_question_time = question_time
        return r.data
    except Exception as e:
        st.error(f"Error generating question: {str(e)}")
        # Return a default question as fallback
        return InterviewQuestion(
            question="Could you describe your most recent technical project and the challenges you faced?",
            time=DEFAULT_QUESTION_TIME_MINUTES
        )


@interviewer_agent.tool
async def generate_evaluation_criteria(ctx: RunContext[InterviewDetails]) -> str:
    """Create a comprehensive evaluation framework for this specific interview."""
    try:
        r = await critic_generator.run(
            "Generate an evaluation criteria matrix for this technical role",
            deps=ctx.deps
        )
        return r.data
    except Exception as e:
        st.error(f"Error generating evaluation criteria: {str(e)}")
        return "Technical knowledge (1-100), Problem-solving (1-100), Communication (1-100), Experience relevance (1-100)"


@interviewer_agent.tool(docstring_format='google')
async def evaluate_response(ctx: RunContext[InterviewDetails], criteria: str, question: str, answer: str) -> str:
    """Evaluate the candidate's answer based on established criteria.

    Args:
        criteria: The evaluation criteria to apply
        question: The question that was asked
        answer: The candidate's response to evaluate

    Returns:
        A detailed evaluation of the response
    """
    try:
        r = await response_evaluator.run(
            f"""Evaluate the candidate's answer according to our criteria:
            
            Evaluation Criteria: {criteria}
            Question: {question}
            Candidate's Answer: {answer}
            
            Provide a detailed assessment focusing on technical accuracy, completeness, and relevance.
            """,
            deps=ctx.deps
        )
        # Store the response in session state for later saving
        if "responses" in st.session_state:
            st.session_state.responses.append(r.data.model_dump())
        return r.data.evaluation
    except Exception as e:
        st.error(f"Error evaluating response: {str(e)}")
        return "Unable to evaluate the response due to a technical issue."


@interviewer_agent.tool
async def evaluate_interview(ctx: RunContext[InterviewDetails]) -> str:
    """Perform a comprehensive evaluation of the entire interview."""
    responses = st.session_state.responses if "responses" in st.session_state else []
    
    try:
        formatted_responses = ""
        for i, response in enumerate(responses):
            formatted_responses += f"Question {i+1}: {response.get('question')}\n"
            formatted_responses += f"Answer: {response.get('answer')}\n"
            formatted_responses += f"Evaluation: {response.get('evaluation')}\n"
            formatted_responses += f"Score: {response.get('score')}\n\n"
        
        r = await summative_evaluator.run(
            f"""Provide a comprehensive evaluation of this interview:
            
            Job Description: {ctx.deps.job_description}
            Candidate Resume: {ctx.deps.candidate_information}
            
            Interview Responses:
            {formatted_responses}
            
            Consider technical skills, problem-solving abilities, and job fit.
            """,
            deps=ctx.deps
        )
        return r.data
    except Exception as e:
        st.error(f"Error in final evaluation: {str(e)}")
        return "Unable to generate a final evaluation due to a technical issue."


@interviewer_agent.tool(docstring_format='google')
async def save_results(ctx: RunContext[InterviewDetails], summary: str) -> str:
    """Save the complete interview results to the database.

    Args:
        summary: The final summative evaluation of the candidate

    Returns:
        Confirmation message about saved data
    """
    try:
        # Initialize database connection if not already done
        if "db" not in st.session_state:
            st.session_state.db = Database()
        
        # Prepare interview data based on the source (URL params or direct input)
        if st.query_params and "candidate_id" in st.query_params and "job_desc_id" in st.query_params:
            interview_data = {
                "candidate_id": st.query_params['candidate_id'],
                "job_desc_id": st.query_params['job_desc_id'],
                "responses": st.session_state.responses,
                "summary": summary,
                "timestamp": time.time()
            }
        else:
            interview_data = {
                "resume": ctx.deps.candidate_information,
                "job_description": ctx.deps.job_description,
                "responses": st.session_state.responses,
                "summary": summary,
                "timestamp": time.time()
            }
        
        # Save to database
        result_id = st.session_state.db.save_interview(interview_data)
        if result_id:
            return f"Interview results saved successfully with ID: {result_id}"
        else:
            return "Failed to save interview results to the database."
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return "Failed to save interview results due to a technical error."


@interviewer_agent.system_prompt
async def get_interview_detail(ctx: RunContext[InterviewDetails]) -> str:
    """Provide context about the interview to the agent."""
    return f"""
    Job Description: 
    {ctx.deps.job_description}
    
    Candidate Information:
    {ctx.deps.candidate_information}
    """


# --- Timer Implementation ---

async def countdown_timer(seconds: int, timer_placeholder):
    """Display a countdown timer in the Streamlit app.
    
    Args:
        seconds: Total seconds for countdown
        timer_placeholder: Streamlit element to display the timer
    """
    end_time = time.time() + seconds
    
    while time.time() < end_time:
        remaining = end_time - time.time()
        mins, secs = divmod(int(remaining), 60)
        timer_placeholder.markdown(f"⏱️ Time remaining: **{mins}:{secs:02d}**")
        await asyncio.sleep(0.5)
    
    # Time's up
    timer_placeholder.markdown("⏰ **Time's up!**")
    return True


# --- Main Application Function ---

async def main():
    """Main application function that handles the Streamlit UI and flow."""
    
    # Set up page config
    st.set_page_config(
        page_title="AI Technical Interviewer",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state variables
    if "interview_details" not in st.session_state:
        st.session_state.interview_details = None
    if "history" not in st.session_state:
        st.session_state.history = []
    if "display_msg" not in st.session_state:
        st.session_state.display_msg = ""
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "current_question_time" not in st.session_state:
        st.session_state.current_question_time = DEFAULT_QUESTION_TIME_MINUTES
    if "db" not in st.session_state:
        st.session_state.db = Database()
    if "interview_in_progress" not in st.session_state:
        st.session_state.interview_in_progress = False
    
    # Page header
    st.title("🤖 AI Technical Interviewer")
    st.markdown("""
    This system conducts a technical interview based on your resume and the job description.
    Answer each question within the time limit for the best assessment.
    """)
    
    # Create sidebar for settings and info
    with st.sidebar:
        st.header("Information")
        st.info("""
        This AI interviewer will:
        - Ask up to 6 technical questions
        - Evaluate your responses
        - Provide feedback and a final assessment
        
        Each question has a time limit. Try to answer before time runs out!
        """)
        
        if "db_connected" in st.session_state and st.session_state.db_connected:
            st.success("✅ Database connected")
        else:
            st.error("❌ Database not connected")
    
    # Get job description and resume - either from URL parameters or manual input
    job_description = None
    resume_file = None

    if st.query_params and "candidate_id" in st.query_params and "job_desc_id" in st.query_params:
        # Load data from database based on URL parameters
        try:
            job_desc = st.session_state.db.get_job_description(st.query_params['job_desc_id'])
            if job_desc:
                job_description = job_desc.get('job_desc')
                st.write("📄 **Job Description loaded from database**")
                
                # Get candidate info
                candidate = st.session_state.db.get_candidate(st.query_params['candidate_id'])
                if candidate and "resume_url" in candidate:
                    file_url = candidate.get("resume_url")
                    resume_file = download_file(file_url)
                    if resume_file:
                        st.write("📄 **Resume loaded from database**")
                    else:
                        st.error("Failed to download resume file")
                else:
                    st.error("Candidate information not found")
            else:
                st.error("Job description not found")
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
    
    # If not using URL parameters or if database retrieval failed, show manual input fields
    if not job_description or not resume_file:
        col1, col2 = st.columns(2)
        
        with col1:
            job_description = st.text_area(
                "📝 Job Description:", 
                height=200,
                placeholder="Paste the job description here..."
            )
            
        with col2:
            resume_file = st.file_uploader(
                "📎 Your Resume (PDF format):", 
                type="pdf",
                help="Upload your resume in PDF format"
            )
    
    # Once we have both job description and resume, proceed with interview
    if job_description and resume_file:
        # Display information about the loaded documents
        with st.expander("📑 Loaded Documents", expanded=False):
            st.markdown("### Job Description")
            st.write(job_description[:500] + "..." if len(job_description) > 500 else job_description)
            
            st.markdown("### Resume")
            resume_text = extract_text_from_pdf(resume_file)
            st.write(resume_text[:500] + "..." if len(resume_text) > 500 else resume_text)
        
        # Initialize interview if not already done
        if not st.session_state.interview_details:
            st.session_state.interview_details = InterviewDetails(
                job_description=job_description,
                candidate_information=resume_text
            )
        
        # Start interview when button is clicked
        if not st.session_state.interview_in_progress:
            if st.button("🚀 Start Interview", type="primary"):
                with st.spinner("Preparing your interview..."):
                    # Initialize the interview
                    result = await interviewer_agent.run(
                        user_prompt="Hello, I'm ready to begin the interview.",
                        deps=st.session_state.interview_details
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    st.session_state.interview_in_progress = True
                    st.session_state.start_time = time.time()
                st.rerun()
        else:
            # Display interview in progress
            st.markdown("### Interview in Progress")
            
            # Display chat history
            chat_container = st.container()
            with chat_container:
                with st.chat_message("assistant", avatar="🤖"):
                    st.write(st.session_state.display_msg)
            
            # Display timer
            timer_placeholder = st.empty()
            
            # Input for candidate's answer
            answer = st.chat_input("Your answer...")
            
            if answer:
                # Display user's message
                with chat_container:
                    with st.chat_message("user", avatar="👤"):
                        st.write(answer)
                
                # Process the answer
                with st.spinner("Evaluating your answer..."):
                    result = await interviewer_agent.run(
                        user_prompt=answer,
                        deps=st.session_state.interview_details,
                        message_history=st.session_state.history
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    st.session_state.start_time = time.time()
                    st.rerun()
            
            # Show the timer while waiting for an answer
            asyncio.create_task(countdown_timer(
                60 * st.session_state.current_question_time, 
                timer_placeholder
            ))


# --- Application Entry Point ---

if __name__ == "__main__":
    asyncio.run(main())
```
