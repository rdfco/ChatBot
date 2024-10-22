import logging

import pandas as pd
from langchain_openai import ChatOpenAI
from openai import OpenAI

from utils.exceptions import CouldNotFindAnswerException, InternalServerErrorException

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configs
gtp_model = "gpt-4o"
api_key = "sk-7Nv8_8hz-4llVHaLZwi0jo3fXedJcbWmEv3bYaaUfKT3BlbkFJjjgj6LPUoAA3zxH5Dp4Poj3ZNJsmPeL9G9qfK2BoYA"
client = OpenAI(api_key=api_key)
model = ChatOpenAI(model=gtp_model, api_key=api_key)
agent_name = "STIM"
assistant = None
# FIXME: The file paths can be changed
# The file paths should be a dictionary with the key as the subject and the value as a list of the file paths
file_paths = {
    "Sponge Iron": [
        "files/Sponge Iron report new version 2.pdf",
        "files/_isic_.csv",
    ],
}

# Get the assistant if exists
for agent in client.beta.assistants.list():
    if agent.name == agent_name:
        assistant = agent
        break

# Creating assistant if not exists
if not assistant:
    # Get the file paths for the search and code interpreter tools based on the file extensions
    vector_stores = []
    code_interpreter_uploaded_files = []

    for subject, files in file_paths.items():
        # Create vector store
        vector_store = client.beta.vector_stores.create(name=subject)
        vector_stores.append(vector_store.id)

        # Upload files
        file_streams = [open(path, "rb") for path in files if path.endswith(".pdf")]
        file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=vector_store.id, files=file_streams
        )

        code_interpreter_files = [
            client.files.create(file=open(path, "rb"), purpose="assistants").id
            for path in files
            if path.endswith(".csv")
        ]
        code_interpreter_uploaded_files.extend(code_interpreter_files)

    # Create assistant
    assistant = client.beta.assistants.create(
        name=agent_name,
        instructions="You are an expert patent analyst. When asked a question, you will answer to the threads' messages based on the uploaded files.",
        tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        model=gtp_model,
        tool_resources={
            "code_interpreter": {"file_ids": code_interpreter_uploaded_files},
            "file_search": {"vector_store_ids": vector_stores},
        },
    )

# Get user input
user_message = input("You > ")

csv_file = pd.read_csv("files/prompts.csv")

# Create a thread
thread = client.beta.threads.create(
    messages=[
        {
            "role": "user",
            "content": f"Get the question of the client, then check it with the prompts (shown in 'prompt' column) and find the similar prompts. If there are more than one similar prompts, select the most similar one. And then extract the response (shown in 'response' column). If there is no similar prompt, then investigate whether it is related to the report or not. If related, extract the response from the report file. If you can not find it from the report, respond it by yourself from public references. If it is not related, return 'No response find'. Client question: {user_message}",
        },
        {
            "role": "user",
            "content": f"""
            Prompts and responses:
            {csv_file}
            """,
        },
    ]
)

# Run the thread
run = client.beta.threads.runs.create_and_poll(
    thread_id=thread.id,
    assistant_id=assistant.id,
    max_prompt_tokens=3000,
)

# Get the final response
while run.status != "completed":
    run = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
    if run.status == "failed":
        logging.error(run)
        print(InternalServerErrorException())

messages = client.beta.threads.messages.list(thread_id=thread.id)
try:
    print("Assistant > ", messages.data[0].content[-1].text.value)
except Exception as e:
    logging.error(e)
    print(CouldNotFindAnswerException())
