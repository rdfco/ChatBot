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

# Our prompts
prompts = [
    """List the top 10 companies with the most patents. 
For each company mention its name properly with its nationality also make a list of universities among these 10 companies 
( write it in a proper format for a professional report)""",
    "What is the patent registration strategy of the top 10 companies with the highest number of patents in different countries?",
    """Which companies hold the most valuable patents
( write the answer in a proper format for a professional report)?""",
    """Which companies do the top 10 companies with the most patents collaborate with the most? 
How many of their collaborators are companies and how many are universities?""",
    """What has been the continuity of activities of the top companies over the past 10 years? 
Which companies have consistently registered patents, and in which companies has patent registration stopped?""",
    """Which companies have registered the most patents in the last 5 years? 
Which of these are not in the top 10 companies by the number of patents but have recently emerged? Which company has the highest growth rate?""",
    """Name the company with the highest patent growth rate in the last 5 years and 
provide a brief explanation about the company and its activities related to the report(AI-Driven Drug Discovery), up to 150 words.""",
    """Name the companies that are not in the top 10 by the number of patents but are in the top 5 in the last 5 years and provide a brief explanation about them.""",
    "Which companies have the most patents pending review for activation, and what is the predicted trend for patent registrations in the next 5 years?",
    """List the pending patents of the companies with the most pending patents and 
specify the focus areas of these patents for each company.""",
    """Based on the registered codes in the main patent clusters, what are the focus areas of patents in this field? 
Provide a general analysis.""",
    """Based on the main cluster in this field, state which cluster the main jurisdictions are focusing on. 
Be sure to compare the USA and China.""",
    """Take the first 5 main groups and explain that this main cluster is one of the 
focus areas and what it explains about""",
    """Based on the registered codes in the main patent clusters, what are the focus areas of patents in this field? 
Provide a general analysis.""",
    """Based on the main cluster in this field, state which cluster the main jurisdictions are focusing on. 
Be sure to compare the USA and China.""",
    """Take the first 5 sub groups and explain that this main cluster is one of the 
focus areas and what it explains about""",
    """Based on sub-group data and the most registered groups, 
what are the overall technology trends in this field? Highlight up to 5 trends.""",
    """Based on technology clusters, introduce the top 5 clusters in the last 5 years with an explanation of each cluster.
 Also, mention which cluster has the highest growth.""",
    """Based on the top 5 technologies of the last 5 years, describe the status of different jurisdictions. 
Specifically, compare the USA and China.""",
    """Take the main patent which is US 2020/0184278 A1 : 
System and Method for Extremely Efficient Image and Pattern Recognition and Artificial Intelligence Platform  
 in the graph, and state which companies it is from and what the general topics are. List them in bullet points.""",
    """Take the top patent based on cited by patent count which is US 2020/0184278 A1 : System and Method for Extremely Efficient Image and Pattern Recognition and Artificial Intelligence Platform   , and explain what this patent with this patent number is about in a maximum of 200 words. Include: 
•  Patent title

•  Registering company

•  Year of registration

•  Explanation about the patent""",
    """Take the top company based on the number of registered patents and provide a brief background about this company. Briefly explain what it does in the context of this report(AI-Driven Drug Discovery):
University of California (UC) with 27 patents""",
    """If this company is a product manufacturer, what are its top products in the market? 
If it is not a product manufacturer and develops production processes, what are the main processes it has and works on?""",
    """Based on the 10 main groups in which this company( California University) has registered the most patents, explain which technology areas the company focuses on.
Provide a general explanation in 200 words.""",
    """Identify the three main areas of focus for this company in the main group and provide a brief explanation for each in bullet points. 
(That means we have three bullets showing the main focus of this company.)""",
    """Based on the priorities of the main group, it seems that this company is targeting which products or functions, and what is likely the company's approach to product and business development focused on?

One of the company's focus areas is on this technology cluster. Explain this technology cluster.""",
    """Based on the 10 sub-groups (sub-clusters) in which this company has registered the most patents, explain which technology areas this company is focusing on. 
Provide a general explanation in 200 words.""",
    """Identify the three main areas of focus for this company in the subgroups and provide a brief explanation for each in bullet points. 
That means we have three bullets showing the main focus of this company.""",
    """Based on the priorities of the sub-groups, it seems that this company is targeting which products or functions, and what is likely the company's approach to product and business development focused on?
One of the company's focus areas is on this technology sub-cluster. Explain this technology sub-cluster.""",
    """Which companies does this company collaborate with to develop its patents? 
Briefly explain who the two main partners are. Also, explain the key collaborating universities.""",
]

# Prompt selector using ChatGPT
prompt_template = PromptTemplate.from_template(
    """I have a list of 'Prompts' related to patents. Please select the one that best matches of the 'Query' and only return the prompt. If none of the prompts match, but it's related to the subject, please ask GPT and return the answer. If it's not related to the subject, just say 'Sorry, I could not find the answer.', otherwise just say 'Ask Assistant'.
    Query: {query}
    
    Prompts:
    {prompts}
    """
)
parser = StrOutputParser()
chain = model | parser
final_prompt = prompt_template.invoke({"prompts": prompts, "query": user_message})
final_user_message = chain.invoke(final_prompt)


logging.info(f"Final user prompt: {final_user_message}")

# If the user query is not related to the subject, raise an exception, otherwise if the user query is related to the subject, ask public GPT, otherwise ask the assistant based on the uploaded files
if final_user_message != "Ask Assistant":
    print(final_user_message)
else:
    # Create a thread
    thread = client.beta.threads.create(
        messages=[
            {
                "role": "user",
                "content": final_user_message,
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
