import asyncio
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
import uuid
import random

import streamlit as st
import logfire
from pydantic import BaseModel, Field
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai import Agent, RunContext

nest_asyncio.apply()

# Load environment variables
load_dotenv(override=True)

model = OpenAIModel(os.getenv("MODEL"), api_key=os.getenv("OPENAI_API_KEY"))
logfire.configure(send_to_logfire='if-token-present')

# model_settings = ModelSettings(
#     temperature=0.75,
#     parallel_tool_calls=False
# )

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["eng_tests"]
question_bank = db["question_bank"]  # New collection for storing question bank


class Question(BaseModel):
    """Details of an interview question to ask a user."""
    question: str = Field(description="The specific question to ask the user")
    question_type: Literal[
        "short_answer", "multiple_choice", "yes_no", "reading_comprehension", "fill_in_blanks", "error_identification", "functional_language"] = Field(
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
    difficulty_level: Literal[
        "beginner", "elementary", "intermediate", "upper-intermediate", "advanced", "proficient"] = Field(
        description="The difficulty level of the question"
    )
    question_id: Optional[str] = Field(
        description="Unique identifier for this question",
        default_factory=lambda: str(uuid.uuid4())
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
    retries=1,
    result_type=Question,
    # model_settings=model_settings,
    system_prompt=(
        "You are a friendly expert interviewer who evaluates the users english proficiency level. "
        "Your goal is to assess a user's English proficiency level. "
        "You must follow these steps:\n"
        "1. Begin the test with opening remarks in Mongolian language.\n"
        "2. Ask questions from the user assessing their vocabulary, grammar, reading comprehension. You can ask the following types of questions:\n"
        " - Very short reading passage with a reading comprehension question\n"
        " - Fill in the blanks\n"
        " - Error identification\n"
        " - Cohesion and coherence\n"
        " - Functional english usage / Real world use of english\n"
        " Your question format must be multiple choice or short answer questions.\n"
        " Provide a short feedback on previous questions and guiding and encouraging words without greetings at the start of your questions.\n"
        "3. After you finish asking questions, use evaluate_test tool and save the result with save_results tool.\n"
        "4. Finish the interview with closing remarks in Mongolian language.\n\n"
        "Guidelines:\n"
        "- Your feedback throughout the test must be in Mongolian language and your questions must be in English.\n"
        "- Ask only one question at a time and wait for the answer.\n"
        "- Start with an intermediate-level question\n"
        "- If the user answers correctly, you must move to a one level harder and complex question highest being up to and including C2 level on cambridge scale\n"
        "- If the user answers correctly to a C2 level question, you must keep asking C2 level question\n"
        "- If the user answers incorrectly, you must move to a one level easier and simpler question lowest being down to and including A1 level on cambridge scale\n"
        # "- The difficulty of each question should be changed according to the cambridge english scale"
        "- Do not ask the same question twice\n"
        "- Do not ask the same type of question twice in a row\n"
        "- Output a JSON and include options in a separate list in the JSON for multiple choice type questions. \n"
        f"- Conclude the test after {os.getenv('Q_CNT')} questions"
    ),
)

summary_evaluator = Agent(
    model=model,
    result_type=Summary,
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
    Take the difficulty of questions into account when evaluating.
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
    if st.query_params and "candidate_id" in st.query_params:
        interview_data = {
            "candidate_id": st.query_params["candidate_id"],
            "responses": st.session_state.responses,
            "summary": summary.model_dump(),
            "session_id": st.session_state.session_id,
            "timestamp": time.time()
        }
    else:
        interview_data = {
            "responses": st.session_state.responses,
            "summary": summary.model_dump(),
            "session_id": st.session_state.session_id,
            "timestamp": time.time()
        }
    collection.insert_one(interview_data)
    st.session_state.finished = True
    return "Interview results saved successfully."


# Improved timer implementation as a hoverable component
def timer_component(timer_placeholder, seconds_remaining):
    minutes = int(seconds_remaining // 60)
    seconds = int(seconds_remaining % 60)

    # Main display with tooltip
    timer_placeholder.markdown(
        f"""
        <div style="position: relative; display: inline-block;">
            <div class="timer-icon" style="cursor: pointer; background-color: {'green' if seconds_remaining > 60 else 'orange' if seconds_remaining > 30 else 'red'}; 
                 padding: 5px 10px; border-radius: 5px; color: white; font-weight: bold; font-size: 40px; text-align: center;">
                ⏱️
            </div>
            <div style="text-align: center; border-radius: 6px; 
                 padding: 5px 10px;">
                Үлдсэн хугацаа: {minutes}:{seconds:02d}
            </div>
        </div>""",
        unsafe_allow_html=True
    )

    # Return True if time is up
    return seconds_remaining <= 0


async def timer_handler(timer_placeholder, seconds):
    end_time = time.time() + seconds

    while time.time() < end_time and not st.session_state.get("answer_submitted", False):
        remaining = end_time - time.time()
        time_up = timer_component(timer_placeholder, remaining)
        if time_up:
            st.session_state.time_up = True
            break

        timer_placeholder.empty()
        with timer_placeholder:
            timer_component(timer_placeholder, remaining)

        await asyncio.sleep(1)

    if not st.session_state.get("answer_submitted", False):
        st.session_state.time_up = True
        st.session_state.messages.append({"role": "user", "content": "Цаг дууссан байна."})
        st.session_state.responses.append({
            "question": st.session_state.display_msg.question,
            "response": "No answer (time expired).",
            "question_type": st.session_state.display_msg.question_type,
            "difficulty_level": st.session_state.display_msg.difficulty_level
        })
        st.rerun()


Agent.instrument_all()


async def main():
    st.set_page_config(
        page_title="Англи хэлний түвшин тогтоох тест",
        page_icon="🎓",
        layout="centered"
    )

    st.title("Англи хэлний түвшин тогтоох тест")

    # Initialize session state variables
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
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
    if "used_questions" not in st.session_state:
        st.session_state.used_questions = []
    if "difficulty_level" not in st.session_state:
        st.session_state.difficulty_level = "intermediate"
    if "previous_type" not in st.session_state:
        st.session_state.previous_type = None
    if "question_count" not in st.session_state:
        st.session_state.question_count = 0
    if "time_up" not in st.session_state:
        st.session_state.time_up = False
    if "answer_submitted" not in st.session_state:
        st.session_state.answer_submitted = False

    # Container for chat history
    chat_container = st.container()

    # Container for timer
    timer_placeholder = st.sidebar.empty()

    # Container for input and controls
    input_container = st.container()

    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Handle test completion
    if st.session_state.finished:
        st.success("Та тестийг амжилттай дуусгалаа! Оролцсон танд баярлалаа.")
        # if st.button("Start New Test"):
        #     # Reset session state for a new test
        #     for key in ["messages", "history", "responses", "display_msg",
        #                 "used_questions", "question_count", "finished"]:
        #         if key in st.session_state:
        #             del st.session_state[key]
        #     st.rerun()
        return

    # Handle time up condition
    if st.session_state.time_up and not st.session_state.answer_submitted:
        with st.spinner("Цаг дууслаа, Дараагийн асуулт руу шилжиж байна..."):
            st.session_state.time_up = False
            result = await tester_agent.run(
                user_prompt="No answer (time expired).",
                message_history=st.session_state.history
            )
            st.session_state.history = result.all_messages()
            st.session_state.display_msg = result.data
            st.session_state.messages.append({"role": "assistant", "content": result.data.question})
            st.session_state.start_time = time.time()
            st.session_state.question_count += 1
            st.session_state.previous_type = st.session_state.display_msg.question_type
            st.rerun()

    # Initialize first question or display current question
    if len(st.session_state.history) == 0:
        with st.spinner("Тестийг бэлдэж байна..."):
            # Get the first question
            first_question = await tester_agent.run(
                user_prompt="Hello, I'm ready for the test.",
            )
            st.session_state.history = first_question.all_messages()
            st.session_state.display_msg = first_question.data
            st.session_state.messages.append({"role": "assistant", "content": first_question.data.question})
            st.session_state.start_time = time.time()
            st.session_state.question_count = 1

            # Start the timer only after the button is clicked
            if st.button("Эхлэх", key="start_test"):
                st.session_state.start_time = time.time()

                # Set up timer task
                st.session_state.current_question_time = first_question.data.time or 2
                asyncio.run(timer_handler(
                    timer_placeholder,
                    60 * st.session_state.current_question_time
                ))
                st.rerun()

    else:
        # Process current question
        with input_container:
            # Display appropriate input method based on question type
            st.session_state.answer_submitted = False

            if st.session_state.display_msg.options:
                options = st.session_state.display_msg.options
                selected_option = st.radio("Select your answer:", options, key="mc_response")
                if st.button("Хариулт илгээх", key="submit_mc"):
                    st.session_state.answer_submitted = True
                    st.session_state.messages.append({"role": "user", "content": selected_option})
                    with st.spinner("Хариултыг дүгнэж байна..."):
                        st.session_state.responses.append({
                            "question": st.session_state.display_msg.question,
                            "response": selected_option,
                            "question_type": st.session_state.display_msg.question_type,
                            "difficulty_level": st.session_state.display_msg.difficulty_level
                        })

                        result = await tester_agent.run(
                            user_prompt=f"My answer is: {selected_option}",
                            message_history=st.session_state.history
                        )
                        st.session_state.history = result.all_messages()
                        st.session_state.display_msg = result.data
                        st.session_state.messages.append({"role": "assistant", "content": result.data.question})
                        st.session_state.start_time = time.time()
                        st.session_state.question_count += 1
                        st.session_state.current_question_time = result.data.time or 2
                        st.session_state.difficulty_level = result.data.difficulty_level

                        # Reset timer
                        asyncio.run(timer_handler(
                            timer_placeholder,
                            60 * st.session_state.current_question_time
                        ))
                        st.rerun()
            else:
                # Text input for short answers
                answer = st.chat_input("Таны хариулт:")
                if answer:
                    st.session_state.answer_submitted = True
                    st.session_state.messages.append({"role": "user", "content": answer})
                    with st.spinner("Хариултыг дүгнэж байна..."):
                        st.session_state.responses.append({
                            "question": st.session_state.display_msg.question,
                            "response": answer,
                            "question_type": st.session_state.display_msg.question_type,
                            "difficulty_level": st.session_state.display_msg.difficulty_level
                        })

                        result = await tester_agent.run(
                            user_prompt=answer,
                            message_history=st.session_state.history
                        )
                        st.session_state.history = result.all_messages()
                        st.session_state.display_msg = result.data
                        st.session_state.messages.append({"role": "assistant", "content": result.data.question})
                        st.session_state.start_time = time.time()
                        st.session_state.question_count += 1
                        st.session_state.current_question_time = result.data.time or 2
                        st.session_state.difficulty_level = result.data.difficulty_level

                        # Reset timer
                        asyncio.run(timer_handler(
                            timer_placeholder,
                            60 * st.session_state.current_question_time
                        ))
                        st.rerun()

            # Display progress indicator
            st.progress(min(st.session_state.question_count / int(os.getenv("Q_CNT")), 1.0))
            st.caption(
                f"Асуулт {st.session_state.question_count}/{os.getenv('Q_CNT')} - Түвшин: {st.session_state.difficulty_level}")

        # Start or continue timer
        if not st.session_state.answer_submitted:
            asyncio.run(timer_handler(
                timer_placeholder,
                max(0, (60 * st.session_state.current_question_time) - (time.time() - st.session_state.start_time))
            ))


if __name__ == "__main__":
    asyncio.run(main())
