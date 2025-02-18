import os
import time
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent, config_list_from_json
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import asyncio
import json


# Load environment variables
load_dotenv()

# Load OpenAI API key
config_list = [
    {
        'model': os.getenv("MONGO_URI"),
        'api_key': os.getenv("OPENAI_API_KEY")
    }
]

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["candidate_interviews"]

# Function to extract text from a PDF (for the resume)
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Function to generate interview questions
def generate_interview_questions(job_description, resume_text):
    # Create an AutoGen assistant agent
    assistant = AssistantAgent(
        name="interview_agent",
        llm_config={"config_list": config_list},
    )

    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    # Define the prompt for generating questions
    prompt = f"""
    You are an AI interviewer. Your task is to generate a set of behavioral, case-based, problem-solving, and multiple-choice questions to assess a candidate's fit for a job based on the job description and their resume.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Generate:
    - 3 behavioral questions
    - 3 case-based questions
    - 3 problem-solving tasks (to be completed with plain text)
    - 3 multiple-choice questions (provide 4 options for each)

    Return a JSON object with 'behavioral', 'case-based', 'problem-solving' and 'multiple-choice' keys and questions as a list.
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    print("LASTMSG: ", assistant.last_message())

    # Retrieve the generated questions
    questions = assistant.last_message()["content"]
    questions = questions.lstrip("```json\n").rstrip("\n```")
    questions_dict = json.loads(questions)
    return questions_dict  # Split into individual questions

# Function to determine if a follow-up question is needed
def needs_follow_up(question, response):
    # Create an AutoGen assistant agent
    assistant = AssistantAgent(
        name="follow_up_agent",
        llm_config={"config_list": config_list},
    )

    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    # Define the prompt for follow-up determination
    prompt = f"""
    You are an AI interviewer. Your task is to determine if a follow-up question is needed based on the candidate's response.

    Question: {question}
    Response: {response}

    Decide if a follow-up question is needed. If yes, provide the follow-up question. If no, simply say "No follow-up needed."
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the follow-up decision
    follow_up = assistant.last_message()["content"]
    return follow_up

# Function to score a response
def score_responses(questions, responses, job_description):
    # Create an AutoGen assistant agent
    assistant = AssistantAgent(
        name="scoring_agent",
        llm_config={"config_list": config_list},
    )

    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    # Define the prompt for scoring
    prompt = f"""
    You are an AI evaluator. Your task is to score the candidate's response to an interview question on a scale of 1-10 based on relevance, depth, and alignment with the job description and then make a final decision on whether to hire the candidate based on their interview scores.

    Job Description:
    {job_description}

    Questions and Responses:
    {"".join([f"Q: {q}, A: {r}" for q, r in zip(questions, responses)])}

    Provide a score (1-10) and a brief explanation for the score.
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the score and explanation
    score_result = assistant.last_message()["content"]
    return score_result

# Function to make a final hiring decision
def make_decision(scores):
    # Create an AutoGen assistant agent
    assistant = AssistantAgent(
        name="decision_agent",
        llm_config={"config_list": config_list},
    )

    # Create a user proxy agent
    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    # Define the prompt for decision-making
    prompt = f"""
    You are an AI hiring manager. Your task is to make a final decision on whether to hire the candidate based on their interview scores.

    Scores:
    {scores}

    Decide whether to hire the candidate. Answer with only Yes or No.
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the decision
    decision = assistant.last_message()["content"]
    return decision

async def print_timer(timer, s):
    while True:
        # Timer
        time_remaining = s - (time.time() - st.session_state.start_time)
        print("TIME: ", time_remaining)
        if time_remaining > 0:
            timer.markdown(f"Time remaining: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Time's up! Moving to the next question.")
            st.session_state.responses.append("No response (time's up)")
            st.session_state.scores.append("0/10 (time's up)")
            st.session_state.current_question_index += 1
            if st.session_state.current_question_index >= len(
                    st.session_state.questions[question_types[st.session_state.current_question_type]]):
                st.session_state.current_question_index = 0
                st.session_state.current_question_type += 1
            st.session_state.start_time = time.time()
            st.rerun()

question_types = ['behavioral', 'case-based', 'problem-solving', 'multiple-choice']

def clear_input():
    st.session_state.text_area_answer = ""
# Streamlit app
def main():
    st.title("AI Interview Agent")

    # Upload job description
    job_description = st.text_area("Paste the Job Description here:", height=200)

    # Upload resume
    resume_file = st.file_uploader("Upload your Resume (PDF):", type="pdf")

    if job_description and resume_file:
        # Extract text from the resume
        resume_text = extract_text_from_pdf(resume_file)

        # Generate interview questions
        questions = generate_interview_questions(job_description, resume_text)

        # Initialize session state for questions, responses, follow-ups, and scores
        if "questions" not in st.session_state:
            st.session_state.questions = questions
        if "questions_asked" not in st.session_state:
            st.session_state.questions_asked = []
        if "responses" not in st.session_state:
            st.session_state.responses = []
        if "follow_ups" not in st.session_state:
            st.session_state.follow_ups = []
        if "scores" not in st.session_state:
            st.session_state.scores = []
        if "current_question_index" not in st.session_state:
            st.session_state.current_question_index = 0
        if "current_question_type" not in st.session_state:
            st.session_state.current_question_type = 0
        if "in_follow_up" not in st.session_state:
            st.session_state.in_follow_up = False
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()
        if "button_disabled" not in st.session_state:
            st.session_state.button_disabled = False

        # Evaluate responses after all questions are answered
        if st.session_state.current_question_type >= len(question_types):
            st.subheader("Interview Completed")
            if st.button("Evaluate Responses"):
                # Score all responses
                scores = score_responses(st.session_state.questions_asked, st.session_state.responses, job_description)
                st.session_state.scores = scores

                # Display scores
                st.subheader("Scores:")
                st.write(scores)

                # Make final decision
                decision = make_decision(scores)
                st.subheader("Final Decision:")
                st.write(decision)

                # Save data to MongoDB
                interview_data = {
                    "resume": resume_text,
                    "job_description": job_description,
                    "questions": st.session_state.questions,
                    "questions_asked": st.session_state.questions_asked,
                    "responses": st.session_state.responses,
                    "scores": st.session_state.scores,
                    "decision": decision,
                }
                collection.insert_one(interview_data)

        # Display current question or follow-up
        if st.session_state.current_question_type < len(question_types):
            if not st.session_state.in_follow_up:
                current_question = st.session_state.questions[question_types[st.session_state.current_question_type]][st.session_state.current_question_index]
                st.subheader(f"Question {st.session_state.current_question_index + 1}:")
                if st.session_state.current_question_type == 3:
                    st.write(current_question['question'])
                else:
                    st.write(current_question)
            else:
                current_question = st.session_state.follow_ups[-1]
                st.subheader("Follow-up Question:")
                st.write(current_question)

            input_answer = st.empty()
            # Input for candidate's response
            if st.session_state.current_question_type == 3:
                options = current_question['options']  # Extract options
                response = st.radio("Select an option:", options)
            else:
                response = st.text_area("Your Answer:", value="", key=f"response_{question_types[st.session_state.current_question_type]}_{st.session_state.current_question_index}_followup_{st.session_state.in_follow_up}")

            if response:
                button_text = st.button("Submit")
            else:
                button_text = st.button("Submit", disabled=True)

            # Button to submit response
            if button_text:
                with st.spinner("Please wait"):
                    if st.session_state.current_question_type == 3:
                        st.session_state.questions_asked.append(current_question['question'])
                    else:
                        st.session_state.questions_asked.append(current_question)
                    st.session_state.responses.append(response)

                    if st.session_state.in_follow_up or st.session_state.current_question_type == 3:
                        follow_up = "No follow-up needed"
                    else:
                        # Check if a follow-up is needed
                        follow_up = needs_follow_up(current_question, response)
                    if "No follow-up needed" not in follow_up:
                        st.session_state.follow_ups.append(follow_up)
                        st.session_state.in_follow_up = True
                    else:
                        st.session_state.current_question_index += 1
                        st.session_state.in_follow_up = False

                    if st.session_state.current_question_index >= len(st.session_state.questions[question_types[st.session_state.current_question_type]]):
                        st.session_state.current_question_index = 0
                        st.session_state.current_question_type += 1

                    st.session_state.start_time = time.time()
                    st.rerun()
            timer = st.empty()
            asyncio.run(print_timer(timer, 60))

if __name__ == "__main__":
    main()