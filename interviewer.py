import os
import time
from io import BytesIO
import streamlit as st
from autogen import AssistantAgent, UserProxyAgent
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import asyncio
import json
import requests
from bson import ObjectId
import re

# Load environment variables
load_dotenv(override=True)
# Load OpenAI API key
config_list = [
    {
        'model': os.getenv("MODEL"),
        'api_key': os.getenv("OPENAI_API_KEY")
    }
]

# MongoDB setup
mongo_uri = os.getenv("MONGO_URI")
client = MongoClient(mongo_uri, server_api=ServerApi('1'))
db = client["interview_db"]
collection = db["candidate_interviews"]
candidates = db["candidates"]
job_descriptions = db["job_desc"]


# Function to extract text from a PDF (for the resume)
def extract_text_from_pdf(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text


# Function to analyze job description and resume to determine question types and counts
def analyze_job_and_resume(job_description, resume_text):
    assistant = AssistantAgent(
        name="analysis_agent",
        llm_config={"config_list": config_list},
    )

    user_proxy = UserProxyAgent(
        name="user_proxy",
        is_termination_msg=lambda x: x.get("content", "") and x.get(
            "content", "").rstrip().endswith("TERMINATE"),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=0,
        code_execution_config=False
    )

    prompt = f"""
    Analyze the following job description and resume to determine the most relevant types of interview questions and the appropriate number of questions for each type.
    Assume it is a 20 minute interview.
    It is not necessary to have all types of questions.

    Job Description:
    {job_description}

    Resume:
    {resume_text}

    Return a JSON object with the following structure:
    {{
        "behavioral": <number of behavioral questions>,
        "case-based": <number of case-based questions>,
        "problem-solving": <number of problem-solving tasks>,
        "multiple-choice": <number of multiple-choice questions>
    }}
    """

    user_proxy.initiate_chat(assistant, message=prompt)

    analysis_result = assistant.last_message()["content"]
    if "TERMINATE" in analysis_result:
        analysis_result = analysis_result.rstrip().rstrip("TERMINATE").rstrip()
    analysis_result = analysis_result.split("```json")[1].lstrip().lstrip("```json").lstrip().split("```")[0].rstrip(
        "```").rstrip()
    return json.loads(analysis_result)


# Function to generate interview questions
def generate_interview_questions(job_description, resume_text, question_counts):
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
    You are an AI interviewer. Your task is to generate a set of behavioral, case-based, problem-solving, and multiple-choice questions in Mongolian language to assess a candidate's fit for a job based on the job description and their resume.
    You should primarily focus on what the candidate stated in their resume to generate responsibility oriented questions.
    Total time to complete all the questions should be under 20 minutes.
    Include a rough estimate of the time needed to answer each question in minutes. 
    For tech related jobs, try to come up with coding related questions involving code snippets and completion and specific frameworks or technology mentioned in the resume.   
    
    Job Description:
    {job_description}

    Resume:
    {resume_text}

    You can choose to generate from the following types of questions:
    - {question_counts['behavioral']} behavioral questions
    - {question_counts['case-based']} case-based questions
    - {question_counts['problem-solving']} problem-solving tasks (to be completed with plain text)
    - {question_counts['multiple-choice']} multiple-choice questions (provide 4 options for each)

    Return a JSON object with the following structure:
    {{
        "behavioral": [{{"question": <the question text>, "time": <estimated completion time in integer minutes>}}],
        "case-based": [{{"question": <the question text>, "time": <estimated completion time in integer minutes>}}],
        "problem-solving": [{{"question": <the question text>, "time": <estimated completion time in integer minutes>}}],
        "multiple-choice": [{{"question": <the question text>, "options": [<options>], "correct": <correct answer>, "time": <estimated completion time in integer minutes>}}],
    }}
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the generated questions
    questions = assistant.last_message()["content"]
    questions = re.sub(',(?=\s*[]])', '', questions)
    questions = re.sub(',(?=\s*[}])', '', questions)
    if "TERMINATE" in questions:
        questions = questions.rstrip().rstrip("TERMINATE").rstrip()
    questions = questions.lstrip().lstrip("```json").lstrip().rstrip("```").rstrip()
    return json.loads(questions)


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

    Question in Mongolian language: {question}
    Response in Mongolian language: {response}

    Decide if a follow-up question is needed.
    If yes, provide only the follow-up question in Mongolian language.
    If no, simply say "No follow-up needed."
    
    Return a JSON object with the following structure:
    {{
        "follow_up_needed": <boolean>,
        "question": <follow-up question in Mongolian language>,
        "time": <estimated completion time in integer minutes>
    }}
    """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the follow-up decision
    follow_up = assistant.last_message()["content"]
    if "TERMINATE" in follow_up:
        follow_up = follow_up.rstrip().rstrip("TERMINATE").rstrip()
    follow_up = follow_up.lstrip().lstrip("```json").lstrip().rstrip("```").rstrip()
    return json.loads(follow_up)


def evaluate_response(question, job_description):
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

    if 'follow_up' in question:
        follow_up_prompt = f"""Question: {question['question']}
            Response: {question['response']}
            Follow-up question: {question['follow_up']}
            Follow-up question response: {question['follow_up_response']}"""
    else:
        follow_up_prompt = f"""Question: {question['question']}
            Response: {question['response']}"""

    # Define the prompt for scoring
    prompt = f"""
        You are an AI evaluator. Your task is to score the candidate's response to an interview question on a scale of 1-100 based on relevance, depth, and alignment with the job description.
        You must respond in Mongolian language.

        Job Description:
        {job_description}

        {follow_up_prompt}

        Provide a score (1-100) and a brief explanation for the score.

        Return a JSON object with the following structure:
        {{
            "score": <candidate's score>,
            "explanation": <explanation>
        }}
        """

    # Initiate the conversation
    user_proxy.initiate_chat(assistant, message=prompt)

    # Retrieve the score and explanation
    score_result = assistant.last_message()["content"]
    if "TERMINATE" in score_result:
        score_result = score_result.rstrip().rstrip("TERMINATE").rstrip()
    score_result = score_result.lstrip().lstrip("```json").lstrip().rstrip("```").rstrip()
    score = json.loads(score_result)

    question['score'] = score['score']
    question['explanation'] = score['explanation']
    st.session_state.scores.append(question)


async def print_timer(job_description, timer, s):
    while True:
        # Timer
        time_remaining = s - (time.time() - st.session_state.start_time)
        if time_remaining > 0:
            timer.markdown(f"Үлдсэн хугацаа: {int(time_remaining // 60)}:{int(time_remaining % 60):02d}")
            await asyncio.sleep(1)
        else:
            st.toast("Цаг дууслаа. Дараагийн асуулт руу шилжлээ.")
            current_question = st.session_state.current_question
            if st.session_state.in_follow_up:
                current_question["follow_up_response"] = "Цагтаа багтаж хариулт өгж чадсангүй."
            else:
                current_question["response"] = "Цагтаа багтаж хариулт өгж чадсангүй."
            evaluate_response(current_question, job_description)
            st.session_state.current_question_index += 1
            if st.session_state.current_question_index >= len(
                    st.session_state.questions[question_types[st.session_state.current_question_type]]):
                st.session_state.current_question_index = 0
                st.session_state.current_question_type += 1
            st.session_state.start_time = time.time()
            st.rerun()


question_types = ['behavioral', 'case-based', 'problem-solving', 'multiple-choice']
question_types_mgl = ['Зан төлөвийн', 'Кэйс', 'Шийдэл олох', 'Сонгох']


def clear_input():
    st.session_state.text_area_answer = ""


# Streamlit app
def main():
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

        # Analyze job description and resume to determine question types and counts
        question_counts = analyze_job_and_resume(job_description, resume_text)

        # Generate interview questions
        questions = generate_interview_questions(job_description, resume_text, question_counts)

        # Initialize session state for questions, responses, follow-ups, and scores
        if "questions" not in st.session_state:
            st.session_state.questions = questions
        if "questions_asked" not in st.session_state:
            st.session_state.questions_asked = []
        if "responses" not in st.session_state:
            st.session_state.responses = []
        if "current_question" not in st.session_state:
            st.session_state.current_question = {}
        if "scores" not in st.session_state:
            st.session_state.scores = []
        if "current_question_index" not in st.session_state:
            st.session_state.current_question_index = 0
        if "current_question_type" not in st.session_state:
            st.session_state.current_question_type = 0
        if "start_time" not in st.session_state:
            st.session_state.start_time = time.time()
        if "button_disabled" not in st.session_state:
            st.session_state.button_disabled = False
        if "stage" not in st.session_state:
            st.session_state.stage = "intro"
        if "in_follow_up" not in st.session_state:
            st.session_state.in_follow_up = False

        if st.session_state.stage == "intro":
            st.subheader("Танд энэ өдрийн мэнд хүргэе.")
            st.write(
                "Бидэнтэй ярилцлага хийхээр цаг гаргасанд баярлалаа. Энэ бол таны ур чадвар, туршлага, мөн манай компанид хэрхэн нийцэхийг ярилцах сайхан боломж юм. Та өөрийгөө тайван байлгаж, өөрийн туршлага, үзэл бодлоо нээлттэйгээр хуваалцаарай.")

            if st.button("Эхлэх"):
                st.session_state.stage = "main"
                st.rerun()

        if st.session_state.stage == "conclusion":
            st.subheader("Амжилттай.")
            st.write(
                "Ярилцлагад цаг зав гарган оролцсон танд баярлалаа! Таны туршлага, ур чадварын талаар сонсох сайхан байлаа. Бид таны мэдээллийг сайтар хянаж, дараагийн шатны талаар удахгүй мэдэгдэнэ. Хэрэв танд нэмэлт асуулт байвал бидэнтэй холбогдож болно. Танд амжилт хүсье! 😊")
            if st.button("Дүгнэх"):
                # Display scores
                st.subheader("Үр дүн:")
                st.write(st.session_state.scores)

        if st.session_state.stage == "main":
            # Evaluate responses after all questions are answered
            if st.session_state.current_question_type >= len(st.session_state.questions):
                st.subheader("Амжилттай асуулдуутад хариулж дууслаа.")
                st.write(
                    "Ярилцлагын үйл явц танд хэр санагдсан бэ? Ямар нэгэн сайжруулах зүйл байвал хуваалцана уу. Таны санал бидэнд ирээдүйд ярилцлагын туршлагыг улам сайжруулахад маш чухал байх болно. Баярлалаа! 😊")
                feedback = st.text_area("Санал хүсэлт")
                if st.button("Дуусгах"):
                    # Save data to MongoDB
                    if st.query_params:
                        interview_data = {
                            "candidate_id": st.query_params['candidate_id'],
                            "job_desc_id": st.query_params['job_desc_id'],
                            "questions": st.session_state.questions,
                            "scores": st.session_state.scores,
                            "feedback": feedback
                        }
                    else:
                        interview_data = {
                            "resume": resume_text,
                            "job_description": job_description,
                            "questions": st.session_state.questions,
                            "scores": st.session_state.scores,
                            "feedback": feedback
                        }
                    collection.insert_one(interview_data)
                    st.session_state.stage = "conclusion"
                    st.rerun()

            # Display current question or follow-up
            if st.session_state.current_question_type < len(question_types) and question_types[st.session_state.current_question_type] in st.session_state.questions:
                timer = st.empty()
                if not st.session_state.in_follow_up:
                    current_question = st.session_state.questions[question_types[st.session_state.current_question_type]][
                        st.session_state.current_question_index]
                    st.session_state.current_question = current_question
                    st.subheader(
                        f"{question_types_mgl[st.session_state.current_question_type]} асуулт {st.session_state.current_question_index + 1}:")
                    st.write(current_question['question'])
                else:
                    current_question = st.session_state.current_question
                    st.subheader("Дэлгэрүүлэх асуулт:")
                    st.write(current_question['follow_up'])

                # input_answer = st.empty()
                # Input for candidate's response
                if st.session_state.current_question_type == 3:
                    options = current_question['options']  # Extract options
                    response = st.radio("Дараах сонголтуудаас сонгоно уу:", options)
                else:
                    response = st.text_area("Таны хариулт:", value="",
                                            key=f"response_{question_types[st.session_state.current_question_type]}_{st.session_state.current_question_index}_followup_{st.session_state.in_follow_up}")

                if response:
                    button_text = st.button("Үргэлжлүүлэх")
                else:
                    button_text = st.button("Үргэлжлүүлэх", disabled=True)

                # Button to submit response
                if button_text:
                    with st.spinner("Түр хүлээнэ үү"):
                        if st.session_state.in_follow_up or st.session_state.current_question_type == 3:
                            follow_up = {"follow_up_needed": False}
                        else:
                            # Check if a follow-up is needed
                            follow_up = needs_follow_up(current_question, response)
                        if follow_up["follow_up_needed"]:
                            st.session_state.in_follow_up = True
                            current_question["follow_up"] = follow_up["question"]
                            current_question["response"] = response
                            current_question["time"] = follow_up["time"]
                            st.session_state.current_question = current_question
                        else:
                            # Score all responses
                            if st.session_state.in_follow_up:
                                current_question["follow_up_response"] = response
                            else:
                                current_question["response"] = response
                            evaluate_response(current_question, job_description)
                            # eval_loop.call_soon_threadsafe(asyncio.async, evaluate_response(current_question, job_description))
                            st.session_state.current_question_index += 1
                            st.session_state.in_follow_up = False

                        if st.session_state.current_question_index >= len(
                                st.session_state.questions[question_types[st.session_state.current_question_type]]):
                            st.session_state.current_question_index = 0
                            st.session_state.current_question_type += 1

                        st.session_state.start_time = time.time()
                        st.rerun()
                asyncio.run(print_timer(job_description, timer, 60 * current_question['time']))
            elif st.session_state.current_question_type < len(question_types):
                st.session_state.current_question_type += 1
                st.rerun()


if __name__ == "__main__":
    main()
