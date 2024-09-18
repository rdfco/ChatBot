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
examples = [
    {"input": "Best patent companies", "output": "Top patent applicants"},
    {"input": "Leading patent owners", "output": "Top Owners"},
    {"input": "Most cited patents", "output": "Highly-Cited Applicants"},
    {
        "input": "Industry collaboration trends",
        "output": "Top Ten Applicants’ collaborations",
    },
    {"input": "Top tech fields", "output": "Top Technologies by Class"},
    {
        "input": "Best innovations in last 5 years",
        "output": "Five Recent dominant technologies",
    },
    {"input": "Most important patents", "output": "Key Patents"},
    {
        "input": "Main themes in Sponge Iron patents",
        "output": "The main themes of patents",
    },
    {"input": "Key patent clusters", "output": "Technology clustering"},
    {"input": "Top inventors in patenting", "output": "Top inventors"},
    {
        "input": "Leading applicants in the last 5 years",
        "output": "Pioneer companies in the last 5 years",
    },
    {"input": "Patent application trends", "output": "Patent Family Analysis"},
    {"input": "Patent market reach", "output": "Patent-Market Coverage"},
    {
        "input": "Patent geographical distribution",
        "output": "Geographical jurisdiction",
    },
    {
        "input": "Recent patent advancements",
        "output": "Top Technologies and Main Trends",
    },
    {
        "input": "Technological efficiency improvements",
        "output": "Efficiency Enhancements",
    },
    {"input": "Emission reduction techniques", "output": "Emission Reduction"},
    {
        "input": "Raw materials utilization advancements",
        "output": "Raw Material Utilization",
    },
    {
        "input": "Process control automations",
        "output": "Process Control and Automation",
    },
    {
        "input": "Energy-saving techniques in patents",
        "output": "Energy Consumption Reduction",
    },
    {
        "input": "Hydrogen-based production expansion",
        "output": "Expansion of Hydrogen-Based DRI Plants",
    },
    {
        "input": "New DRI projects",
        "output": "New DRI Projects Underway to Meet Growing Demand",
    },
    {"input": "Patent continuity trends", "output": "Top applicant activity"},
    {
        "input": "Emerging business strategies",
        "output": "Pioneer companies in the last 5 years",
    },
    {
        "input": "Top patent applicant network",
        "output": "The strongest cooperation networks",
    },
    {"input": "Pending patent insights", "output": "Pending patents"},
    {
        "input": "Main processes and market trends",
        "output": "Main Processes and Market Trends",
    },
    {
        "input": "Market share of major companies",
        "output": "Market Share of Main Producers",
    },
    {"input": "Geographical market analysis", "output": "Market at a Glance"},
    {
        "input": "Technological development hot spots",
        "output": "Top Technologies by Sub-Class",
    },
    {"input": "Patent process advancements", "output": "Top Processes"},
    {
        "input": "Mergers and acquisitions in patents",
        "output": "Merge and Acquisitions",
    },
    {"input": "Key patenting activities", "output": "Top applicant clustering"},
    {"input": "Focused technology fields", "output": "Focused industries"},
    {
        "input": "Critical technology segments",
        "output": "Top Technologies by Main-Group",
    },
    {"input": "Sub-group technology trends", "output": "Top Technologies by Sub-Group"},
    {
        "input": "Patent landscape highlights",
        "output": "Patent Landscape Report at a Glance",
    },
    {
        "input": "Recent technology innovations",
        "output": "Five key technology areas in last 5 years",
    },
    {"input": "Significant patents by citations", "output": "Key patents"},
    {"input": "Top patenting countries", "output": "Global patent registrations"},
    {
        "input": "Emerging market trends in patents",
        "output": "Recent market trends in patents",
    },
    {
        "input": "Yearly patent application trends",
        "output": "Yearly patent applications",
    },
    {"input": "Patent examination statistics", "output": "Patent examination process"},
    {
        "input": "Process heat recovery innovations",
        "output": "Improved heat recovery systems",
    },
    {
        "input": "Top furnace designs in patenting",
        "output": "Top furnace designs innovations",
    },
    {
        "input": "Significant collaborative efforts",
        "output": "Key collaborative networks",
    },
    {"input": "Main applicants activity analysis", "output": "Main applicant activity"},
    {
        "input": "Most active regions in patenting",
        "output": "Geographical jurisdiction in patents",
    },
    {"input": "Patent innovation trends", "output": "Innovation trends in patents"},
    {
        "input": "New emerging technologies",
        "output": "Emerging technologies in patents",
    },
    {
        "input": "Advanced monitoring systems in patents",
        "output": "Advanced monitoring systems",
    },
    {
        "input": "Energy efficiency trends in patents",
        "output": "Energy efficiency trends",
    },
    {
        "input": "Technological advancements in DRI",
        "output": "Technological advancements in DRI",
    },
    {"input": "Patent-based technology outlook", "output": "Patent technology outlook"},
    {
        "input": "New product development based on patents",
        "output": "New product development strategies",
    },
    {
        "input": "Future technology trends in patents",
        "output": "Future technology trends",
    },
    {"input": "Leading patenting methods", "output": "Leading patenting methods"},
    {
        "input": "Sustainable technologies in patents",
        "output": "Sustainable technologies trends",
    },
    {
        "input": "Automation in patent processes",
        "output": "Automation in patent processes",
    },
    {
        "input": "Process improvements in patenting",
        "output": "Process improvements in patents",
    },
    {
        "input": "Alternate materials in patents",
        "output": "Alternate materials utilization",
    },
    {
        "input": "Top patent applicants in specific sectors",
        "output": "Top sector-specific patent applicants",
    },
    {"input": "Significant growth areas", "output": "Patent application growth areas"},
    {
        "input": "Market trends by patent analysis",
        "output": "Market trends by patent analysis",
    },
    {
        "input": "Significant challenges in patenting",
        "output": "Challenges in patenting efforts",
    },
    {
        "input": "Notable technological leaders",
        "output": "Technological leaders in patenting",
    },
    {"input": "Technological breakthroughs", "output": "Breakthroughs in patenting"},
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
