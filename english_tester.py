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

model = OpenAIModel(os.getenv("MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
logfire.configure(send_to_logfire='if-token-present')

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["eng_tests"]

class Question(BaseModel):
    """Details of an interview question to ask a user."""
    question: str = Field(description="The specific question to ask the user")
    question_type: Literal["short_answer", "multiple_choice", "yes_no"] = Field(
        description="The type of question format"
    )
    time: Optional[int] = Field(description="The time required to answer the question in minutes", default=2)
    options: Optional[List[str]] = Field(
        description="Answer options for a multiple choice question",
        default=None
    )
    skill_assessed: Optional[str] = Field(
        description="The specific skill being assessed"
    )
    difficulty_level: Literal["beginner", "elementary", "intermediate", "upper-intermediate", "advanced", "proficient"] = Field(
        description="The difficulty level of the question"
    )

class Summary(BaseModel):
    """Structure for the overall interview assessment."""
    overall_score: int = Field(
        description="Aggregate numerical score representing the user's overall test performance on a scale from 1 to 100, where higher scores indicate stronger performance.",
        ge=1,
        le=100
    )
    summary: str = Field(description="The summary of the user's overall test performance")
    level: str = Field(description="The proficiency level of the user's overall test performance")

tester_agent = Agent(
    model=model,
    retries=2,
    result_type=Question,
    system_prompt=(
        "You are a friendly expert interviewer who evaluates the users english proficiency level. "
        "Your goal is to assess a user's English proficiency level. "
        "You must follow these steps:\n"
        "1. Begin the test with opening remarks in Mongolian language.\n"
        "2. Ask questions from the user assessing their vocabulary, grammar, reading comprehension. You can ask the following types of questions:\n"
        " - Multiple choice questions\n"
        " - Short answer\n"
        " - Very short reading passage + Question\n"
        " - Fill in the blanks"
        "3. After you finish asking questions, use evaluate_test tool and save the result with save_results tool.\n"
        "4. Finish the interview with closing remarks in Mongolian language.\n\n"
        "Guidelines:\n"
        # "- Interact with the user in Mongolian language.\n"
        "- Your feedback throughout the test must be in Mongolian language and your questions must be in English.\n"
        "- Ask only one question at a time and wait for the answer.\n"
        "- Start with an intermediate-level question\n"
        "- If the user answers correctly, move to a harder question according to the cambridge english scale\n"
        "- If the user answers incorrectly, move to an easier question according to the cambridge english scale\n"
        "- Do not ask the same question twice\n"
        "- Do not ask the same type of question twice in a row\n"
        "- Conclude the test after 12 questions"
    ),
)

summary_evaluator = Agent(
    model=model,
    result_type=Summary,
    retries=5,
    system_prompt=(
        "You are an expert at providing comprehensive assessments. "
        "Create a detailed summary that evaluates the user's demonstrated skills."
        "\nAvoid generic assessments that could apply to anyone."
        "\nYou must respond in Mongolian language."
    )
)

@tester_agent.tool
async def evaluate_test(ctx) -> Summary:
    """Provide a comprehensive assessment of the entire test."""
    prompt = """
    Create a comprehensive summary evaluation of the entire test.

    All Responses: {responses}

    Provide a detailed assessment of how well the user performed.
    Assess the english proficiency level of the user against the Cambridge English Scale.
    Format your output as a JSON object containing overall_score, summary, and proficiency_level.

    """

    r = await summary_evaluator.run(
        prompt.format(
            responses=st.session_state.responses if "responses" in st.session_state else []
        ),
        deps=ctx.deps
    )
    return r.data


@tester_agent.tool
async def save_results(ctx, summary: Summary) -> str:
    """Save the test results and final assessment."""
    if st.query_params:
        interview_data = {
            "candidate_id": st.query_params.candidate_id,
            "responses": st.session_state.responses,
            "summary": summary.model_dump()
        }
    else:
        interview_data = {
            "responses": st.session_state.responses,
            "summary": summary.model_dump()
        }
    collection.insert_one(interview_data)
    st.session_state.finished = True
    return "Interview results saved successfully."


async def print_timer(timer, s):
    while True:
        time_remaining = s - (time.time() - st.session_state.start_time)
        if time_remaining > 0:
            timer.markdown(f"Time remaining: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Time's up. Moving to the next question.")
            st.session_state.messages.append({"role": "user", "content": "No answer."})
            with st.spinner("Please wait..."):
                st.session_state.responses.append(
                    {"question": st.session_state.display_msg.question, "response": "No answer."})
                result = await tester_agent.run(
                    user_prompt="No answer.",
                    message_history=st.session_state.history
                )
                st.session_state.history = result.all_messages()
                st.session_state.display_msg = result.data
                st.session_state.messages.append({"role": "assistant", "content": result.data.question})
                st.session_state.start_time = time.time()
                st.rerun()


Agent.instrument_all()


async def main():
    st.title("English proficiency test")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "history" not in st.session_state:
        st.session_state.history = []
    if "responses" not in st.session_state:
        st.session_state.responses = []
    if "display_msg" not in st.session_state:
        st.session_state.display_msg = {}
    if "start_time" not in st.session_state:
        st.session_state.start_time = time.time()
    if "current_question_time" not in st.session_state:
        st.session_state.current_question_time = 2
    if "finished" not in st.session_state:
        st.session_state.finished = False

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])


    if len(st.session_state.history) == 0:
        with st.spinner("Preparing the test..."):
            result = await tester_agent.run(
                user_prompt="Hello, I'm ready for the test.",
            )
            st.session_state.history = result.all_messages()
            st.session_state.display_msg = result.data
            st.session_state.messages.append({"role": "assistant", "content": result.data.question})

        if st.button("Start Interview"):
            st.session_state.start_time = time.time()

    else:
        timer = st.empty()

        st.session_state.start_time = time.time()
        if st.session_state.display_msg.options:
            options = st.session_state.display_msg.options
            selected_option = st.radio("Select your answer:", options, key="mc_response")
            if st.button("Submit Answer"):
                st.session_state.messages.append({"role": "user", "content": selected_option})
                with st.spinner("Evaluating your response..."):
                    st.session_state.responses.append({"question": st.session_state.display_msg.question, "response": selected_option})
                    result = await tester_agent.run(
                        user_prompt=f"My answer is: {selected_option}",
                        message_history=st.session_state.history
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    print(st.session_state.display_msg)
                    st.session_state.messages.append({"role": "assistant", "content": result.data.question})
                    st.session_state.start_time = time.time()
                    st.rerun()
        else:
            answer = st.chat_input("Your answer:")
            if answer:
                st.session_state.messages.append({"role": "user", "content": answer})
                with st.spinner("Evaluating your response..."):
                    st.session_state.responses.append(
                        {"question": st.session_state.display_msg.question, "response": answer})
                    result = await tester_agent.run(
                        user_prompt=answer,
                        message_history=st.session_state.history
                    )
                    st.session_state.history = result.all_messages()
                    st.session_state.display_msg = result.data
                    st.session_state.messages.append({"role": "assistant", "content": result.data.question})
                    st.session_state.start_time = time.time()
                    st.rerun()

        # Show timer
        asyncio.run(print_timer(timer, 60 * st.session_state.current_question_time))

if __name__ == "__main__":
    asyncio.run(main())
