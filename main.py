from langchain_community.example_selectors import NGramOverlapExampleSelector
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI

# Configs
gtp_model = "gpt-4o"
api_key = "sk-7Nv8_8hz-4llVHaLZwi0jo3fXedJcbWmEv3bYaaUfKT3BlbkFJjjgj6LPUoAA3zxH5Dp4Poj3ZNJsmPeL9G9qfK2BoYA"
client = OpenAI(api_key=api_key)
model = ChatOpenAI(model=gtp_model, api_key=api_key)
agent_name = "STIM"
vector_store_name = "Sponge Iron"
assistant = None
file_paths = ["files/Sponge Iron report new version 2.pdf", "files/_isic_.csv"]

# Get the assistant if exists
for agent in client.beta.assistants.list():
    if agent.name == agent_name:
        assistant = agent
        break

# Creating assistant if not exists
if not assistant:
    # Get the file paths for the search and code interpreter tools based on the file extensions
    search_files = []
    code_interpreter_files = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            search_files.append(file_path)
        elif file_path.endswith(".csv"):
            code_interpreter_files.append(file_path)

    # Create vector store
    vector_store = client.beta.vector_stores.create(name="Sponge Iron")

    # Upload files
    file_streams = [open(path, "rb") for path in search_files]
    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
        vector_store_id=vector_store.id, files=file_streams
    )
    code_interpreter_uploaded_files = [
        client.files.create(file=open(path, "rb"), purpose="assistants").id
        for path in code_interpreter_files
    ]

    # Create assistant
    assistant = client.beta.assistants.create(
        name=agent_name,
        instructions="You are an expert patent analyst. Use you knowledge base to answer questions about patents.",
        tools=[{"type": "file_search"}, {"type": "code_interpreter"}],
        model=gtp_model,
        tool_resources={
            "code_interpreter": {"file_ids": code_interpreter_uploaded_files},
            "file_search": {"vector_store_ids": [vector_store.id]},
        },
    )

# Get user input
user_message = input("You > ")

# Select the final prompt based on the user input
example_prompt = PromptTemplate(
    input_variables=["input", "output"],
    template="Input: {input}\nOutput: {output}",
)
examples = [
    {"input": "Best companies", "output": "Top applicants"},
]
example_selector = NGramOverlapExampleSelector(
    examples=examples,
    example_prompt=example_prompt,
    threshold=-1.0,
)
dynamic_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=example_prompt,
    prefix="Give the similar sentence to the user query.",
    suffix="Input: {query}\nOutput:",
    input_variables=["query"],
)
parser = StrOutputParser()
chain = dynamic_prompt | model | parser
final_prompt = chain.invoke({"query": user_message})

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
if run.status == "completed":
    messages = client.beta.threads.messages.list(thread_id=thread.id)
    print(messages[-1].content)
else:
    print(run.status)
