import logging

from langchain_chroma import Chroma
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    ConfigurableFieldSpec,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Configs
gtp_model = "gpt-3.5-turbo"
api_key = "sk-7Nv8_8hz-4llVHaLZwi0jo3fXedJcbWmEv3bYaaUfKT3BlbkFJjjgj6LPUoAA3zxH5Dp4Poj3ZNJsmPeL9G9qfK2BoYA"
model = ChatOpenAI(model=gtp_model, api_key=api_key)

csv_loader = CSVLoader("files/prompts.csv")
csv_data = csv_loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
csv_splits = text_splitter.split_documents(csv_data)


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


csv_vectorstore = Chroma.from_documents(
    documents=csv_splits, embedding=OpenAIEmbeddings(api_key=api_key)
)


csv_retriever = csv_vectorstore.as_retriever(search_type="mmr")

pdf_loader1 = PyPDFLoader("files/Sponge Iron report new version 2.pdf")
pdf_loader2 = PyPDFLoader("files/AI and Bio Slides Ver2.pdf")

pdf_docs = pdf_loader1.load() + pdf_loader2.load()

pdf_splits = text_splitter.split_documents(pdf_docs)

pdf_vectorstore = Chroma.from_documents(
    documents=pdf_splits, embedding=OpenAIEmbeddings(api_key=api_key)
)

pdf_retriever = pdf_vectorstore.as_retriever()


RAG_TEMPLATE = """
        You are a patent analyst.
        Use following steps to answer the 'client question':
            1. Receive the "client question."
            2. Compare it against the prompts in the prompts section.
            3. If multiple similar prompts are found, select the most relevant one.
            4. Extract the corresponding answer from the "response" column (in the prompts section of the most similar prompt).
            5. If no similar prompt is found, check whether the question is related to the report.
                5.1 If related, extract the answer from the report.
                5.2 If no information is available in the report, provide an answer using public references.
            6. If the question is unrelated, return "No response found" without answering from public references.

            <prompts>
            {prompts}
            </prompts>

            'client question':
            {question}

            <report>
            {report}
            </report>
        """

rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)

qa_chain = (
    RunnablePassthrough.assign(
        prompts=lambda x: format_docs(csv_retriever.invoke(x["question"]))
    )
    | RunnablePassthrough.assign(
        report=lambda x: format_docs(pdf_retriever.invoke(x["question"]))
    )
    | rag_prompt
    | model
    | StrOutputParser()
)


def get_session_history(user_id: str, conversation_id: str):
    return SQLChatMessageHistory(f"{user_id}--{conversation_id}", "sqlite:///memory.db")


with_message_history = RunnableWithMessageHistory(
    qa_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
    history_factory_config=[
        ConfigurableFieldSpec(
            id="user_id",
            annotation=str,
            name="User ID",
            description="Unique identifier for the user.",
            default="",
            is_shared=True,
        ),
        ConfigurableFieldSpec(
            id="conversation_id",
            annotation=str,
            name="Conversation ID",
            description="Unique identifier for the conversation.",
            default="",
            is_shared=True,
        ),
    ],
)

user_message = input("You > ")


result = with_message_history.invoke(
    {"question": user_message},
    config={"configurable": {"user_id": "1", "conversation_id": "1"}},
)

print("Assistant > ", result)
