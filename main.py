import logging

from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
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
    "Sponge Iron": ["files/Sponge Iron report new version 2.pdf", "files/_isic_.csv"],
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
        instructions="You are an expert patent analyst. Use you knowledge base to answer questions about patents.",
        tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        model=gtp_model,
        tool_resources={
            "code_interpreter": {"file_ids": code_interpreter_uploaded_files},
            "file_search": {"vector_store_ids": vector_stores},
        },
    )

# Get user input
user_message = input("You > ")

# Select the final prompt based on the user input
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
# FIXME: The examples should be changed based on the subject and the given examples
examples = [
    {"input": "Best patent companies", "output": "Top patent applicants"},
]
example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=-1.0,
)
subjects = ", ".join([f"'{subject}'" for subject in list(file_paths.keys())])
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix=f"Give the similar sentence to the user query. If you can't find any and the query is not related to one of the [{subjects}] subjects, please just write 'Can not find any', but if the query is related, please just write 'Ask GPT'",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"],
)
parser = StrOutputParser()
chain = dynamic_prompt | model | parser
final_prompt = chain.invoke({"query": user_message})
logging.info(f"Final prompt: {final_prompt}")

# If the user query is not related to the subject, raise an exception, otherwise if the user query is related to the subject, ask public GPT, otherwise ask the assistant based on the uploaded files
if final_prompt == "Can not find any":
    print(CouldNotFindAnswerException())
elif final_prompt == "Ask GPT":
    chain = model | parser
    print(chain.invoke(user_message))
else:
    # Create a thread
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": final_prompt,
            }
        ]
    )

    # Run the thread
    run = client.beta.threads.runs.create_and_poll(
        thread_id=thread.id,
        assistant_id=assistant.id,
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
