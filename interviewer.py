import asyncio
from PyPDF2 import PdfReader
import streamlit as st
from typing import List, Optional
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
# 'if-token-present' means nothing will be sent (and the example will work) if you don't have logfire configured
logfire.configure(send_to_logfire='if-token-present')

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["candidate_interviews"]
candidates = db["candidates"]
job_descriptions = db["job_desc"]


class InterviewQuestion(BaseModel):
    """Details of an interview question to ask a candidate."""
    question: str
    time: int = Field(description="The time required to answer the question in minutes")
    options: Optional[List[str]] = Field(description="Answer options for a multiple choice question")



# Function to extract text from a PDF (for the resume)
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


class InterviewDetails(BaseModel):
    """Structure for an interview session with a candidate."""

    job_description: str
    candidate_information: str

class InterviewResponse(BaseModel):
    question: str
    answer: str
    evaluation: str
    score: int

interviewer_agent = Agent(
    model=model,
    deps_type=InterviewDetails,
    retries=10,
    system_prompt=(
        "I want you to act as an interviewer strictly following the guideline in the current conversation. "
        "Use the generate_evaluation_criteria tool to generate an evaluation criteria in the beginning and follow it. "
        "Candidate has no idea what the guideline is. "
        "Ask me questions and wait for my answers. Do not write explanations. "
        "Use the generate_question tool to generate a question. "
        "I want you to only reply as an interviewer. "
        "Do not write all the conversation at once. "
        "Do not ask the same question twice. "
        "Do not ask behavioral questions, only ask job specific questions. "
        "Use the evaluate_response tool to evaluate my answer when you receive my input. "
        "If there is an error, point it out. "
        "Finish the interview when you already asked 6 questions or you think it has been 10 minutes. "
        "Use the evaluate_interview tool to perform a summative evaluation of the whole interview at the end. "
        "Use save_results tool to save the list of questions, answers and evaluations after finishing."
    ),
)

question_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewQuestion,
    system_prompt=(
        "You are a helpful assistant to an interviewer interviewing a candidate. "
        "Generate a question to ask a candidate based on criteria matrix, job description and history. "
        "You have the capability to ask yes/no, multiple choice questions. "
        "Focus heavily on technical skills. "
        "Ask question like a real person, only one question at a time. "
        "Do not ask the same question. "
        "Do not repeat the question. "
        "Do ask only one follow-up questions if necessary."
    )
)

critic_generator = Agent(
    model=model,
    deps_type=InterviewDetails,
    system_prompt=(
        "You are a professional interviewer critic."
        "Define a matrix on how to evaluate and score answers to an interview question based on job description and candidate resume. "
        "Define dimensions vertically and horizontally. "
        "For example: typing speed, answer cohesiveness, answer quality, answer accuracy."
    )
)

response_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    result_type=InterviewResponse,
    system_prompt=(
        "You are a professional interviewer evaluating a response to a question."
    )
)

summative_evaluator = Agent(
    model=model,
    deps_type=InterviewDetails,
    system_prompt=(
        "You are a professional interviewer evaluating an interview results in a summative way."
    )
)

@interviewer_agent.tool
async def generate_question(ctx: RunContext[InterviewDetails]) -> InterviewQuestion:
    r = await question_generator.run(
        "Please generate a question",
        deps=ctx.deps
    )
    st.session_state.current_question_time = r.data.time
    return r.data

@interviewer_agent.tool
async def generate_evaluation_criteria(ctx: RunContext[InterviewDetails]) -> str:
    r = await critic_generator.run(
        "Please generate an evaluation criteria",
        deps=ctx.deps
    )
    return r.data

@interviewer_agent.tool(docstring_format='google')
async def evaluate_response(ctx: RunContext[InterviewDetails], criteria, question, answer) -> str:
    """Evaluate the answer to the question based on the evaluation criteria

    Args:
        criteria: response evaluation criteria
        question: the question you asked the candidate
        answer: the answer you are evaluating
    """
    r = await response_evaluator.run(
        f"""Evaluate current answer according to the criteria
        Criteria: {criteria},
        Question: {question},
        Answer: {answer}""",
        deps=ctx.deps
    )
    st.session_state.responses.append(r.data.model_dump())
    return r.data
@interviewer_agent.tool
async def evaluate_interview(ctx: RunContext[InterviewDetails]) -> str:
    r = await response_evaluator.run(
        "Evaluate the whole interview based on the criteria",
        deps=ctx.deps
    )
    return r.data

@interviewer_agent.tool(docstring_format='google')
async def save_results(ctx: RunContext[InterviewDetails], summary: str) -> str:
    """Save the interview and final summative evaluation

    Args:
        summary: the final summative evaluation
    """
    if st.query_params:
        interview_data = {
            "candidate_id": st.query_params['candidate_id'],
            "job_desc_id": st.query_params['job_desc_id'],
            "responses": st.session_state.responses,
            "summary": summary
        }
    else:
        interview_data = {
            "resume": st.session_state.interview_details.candidate_information,
            "job_description": st.session_state.interview_details.job_description,
            "responses": st.session_state.responses,
            "summary": summary
        }
    collection.insert_one(interview_data)

@interviewer_agent.system_prompt
async def get_interview_detail(ctx: RunContext[InterviewDetails]) -> str:
    return f"Interview details: {ctx.deps.model_dump()}"

async def print_timer(job_description, timer, s):
    while True:
        # Timer
        time_remaining = s - (time.time() - st.session_state.start_time)
        if time_remaining > 0:
            timer.markdown(f"Үлдсэн хугацаа: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Цаг дууслаа. Дараагийн асуулт руу шилжлээ.")
            with st.spinner("Түр хүлээнэ үү"):
                result = await interviewer_agent.run(
                    user_prompt="Time's up, moving on to the next question.",
                    deps=st.session_state.interview_details,
                    message_history=st.session_state.history
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data
                st.session_state.start_time = time.time()
                st.rerun()

async def main():
    st.title("AI Interview Agent")

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
                print(f"File URL: {file_url}")

                # Download the file directly from the URL
                response = requests.get(file_url)

                if response.status_code == 200:
                    # Use BytesIO to handle the file in memory
                    resume_file = BytesIO(response.content)
                else:
                    print(f"Failed to download the file. Status code: {response.status_code}")
                    st.toast(f"Файл татахад алдаа гарлаа {response.status_code}")
            else:
                print("No document found with the specified criteria.")
                st.toast("Файл олдсонгүй")
        except Exception as e:
            print(f"An error occurred: {e}")
            st.toast(f"Алдаа гарлаа {e}")

    if not st.query_params and not (job_description and resume_file):
        # Upload job description
        job_description = st.text_area("Ажлын тодорхойлолт оруулна уу:", height=200)

        # Upload resume
        resume_file = st.file_uploader("Өөрийн CV-ээ оруулна уу (PDF):", type="pdf")

    if job_description and resume_file:
        # Extract text from the resume
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
            st.session_state.current_question_time = 1

        if len(st.session_state.history) == 0:
            with st.spinner("Түр хүлээнэ үү"):
                result = await interviewer_agent.run(
                    user_prompt="Hello, Let's start the interview.",
                    deps=st.session_state.interview_details
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data
            if st.button("Start"):
                st.rerun()
        else:
            timer = st.empty()
            with st.chat_message("assistant"):
                st.write(st.session_state.display_msg)
            answer = st.chat_input("Your answer:")

            if answer:
                with st.spinner("Түр хүлээнэ үү"):
                    result = await interviewer_agent.run(
                        user_prompt=answer,
                        deps=st.session_state.interview_details,
                        message_history=st.session_state.history
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    st.session_state.start_time = time.time()
                    st.rerun()
            asyncio.run(print_timer(job_description, timer, 60 * st.session_state.current_question_time))

if __name__ == "__main__":
    asyncio.run(main())
