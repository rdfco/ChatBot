import csv
import logging

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configs
gpt_model = "gpt-4o"
api_key = "sk-7Nv8_8hz-4llVHaLZwi0jo3fXedJcbWmEv3bYaaUfKT3BlbkFJjjgj6LPUoAA3zxH5Dp4Poj3ZNJsmPeL9G9qfK2BoYA"
client = OpenAI(api_key=api_key)
model = ChatOpenAI(model=gpt_model, api_key=api_key)

# Get user input
user_message = input("You > ")

prompts = ""
with open("files/prompts.csv", "r") as file:
    csvFile = csv.reader(file)
    for i, lines in enumerate(csvFile):
        prompts += f"{i}\t{lines[0]}\t{lines[1]}\n"


RAG_TEMPLATE = f"""
You are a expert patent analyst.
Get the 'client question', then check it with the prompts (shown in 'prompt' column) and find the similar prompts. If there are more than one similar prompts, select the most similar one. And then extract the response (shown in 'response' column). If there is no similar prompt, then investigate whether it is related to the report or not. If related, extract the response from the report. If you can not find it from the report, respond it by yourself from public references. If it is not related, return 'No response find' and don't answer it with public references.

client question:
{{question}}

<report>
{{report}}
</report>

<prompts>
{prompts}
</prompts>
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

file_path = "files/Sponge Iron report new version 2.pdf"
loader = PyPDFLoader(file_path)
data = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

local_embeddings = OpenAIEmbeddings(openai_api_key=api_key)

vector_store = Chroma.from_documents(documents=all_splits, embedding=local_embeddings)

retriever = vector_store.as_retriever()


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


qa_chain = (
    {
        "report": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | rag_prompt
    | model
    | StrOutputParser()
)

result = qa_chain.invoke(user_message)
print(result)
