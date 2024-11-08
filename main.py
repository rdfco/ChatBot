import logging
import uuid

from langchain_chroma import Chroma
from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.chat_message_histories import SQLChatMessageHistory
from langchain_community.document_loaders import CSVLoader, PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import ConfigurableFieldSpec, RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class Assistant:
    _gtp_model = "gpt-3.5-turbo"
    _api_key = "sk-7Nv8_8hz-4llVHaLZwi0jo3fXedJcbWmEv3bYaaUfKT3BlbkFJjjgj6LPUoAA3zxH5Dp4Poj3ZNJsmPeL9G9qfK2BoYA"
    _model = ChatOpenAI(model=_gtp_model, api_key=_api_key)
    _pdf_file_ids = {}
    _csv_file_ids = {}

    def __init__(self):
        self.create_csv_vectorstore()
        self.create_pdf_vectorstore()
        self.create_rag_prompt()
        self.initial_qa_chain()

    def create_csv_vectorstore(self):
        csv_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(api_key=self._api_key)
        )

        self.csv_vectorstore = csv_vectorstore

    def create_pdf_vectorstore(self):
        pdf_vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(api_key=self._api_key)
        )

        self.pdf_vectorstore = pdf_vectorstore

    def upload_csv_files(self, file_paths: list[str]):
        csv_splits = []
        for file_path in file_paths:
            csv_loader = CSVLoader(file_path)
            csv_data = csv_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=0
            )
            csv_splits.extend(text_splitter.split_documents(csv_data))
            for i, _ in enumerate(csv_splits):
                self._csv_file_ids[file_path + f"#{i}"] = uuid.uuid4().hex

        self.csv_vectorstore.add_documents(
            documents=csv_splits, ids=list(self._csv_file_ids.values())
        )

    def upload_pdf_files(self, file_paths: list[str]):
        pdf_splits = []
        for file_path in file_paths:
            pdf_loader = PyPDFLoader(file_path)
            pdf_data = pdf_loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500, chunk_overlap=0
            )
            pdf_splits.extend(text_splitter.split_documents(pdf_data))
            for i, _ in enumerate(pdf_splits):
                self._pdf_file_ids[file_path + f"#{i}"] = uuid.uuid4().hex

        self.pdf_vectorstore.add_documents(
            documents=pdf_splits, ids=list(self._pdf_file_ids.values())
        )

    def delete_csv_files(self, ids: list[str]):
        self.csv_vectorstore.delete(ids=ids)

    def delete_pdf_files(self, ids: list[str]):
        self.pdf_vectorstore.delete(ids=ids)

    def create_rag_prompt(self):
        RAG_TEMPLATE = """
        You are a patent analyst.
        Use following steps to answer the 'client question':
            1. Receive the "client question."
            2. Compare it against the prompts in the prompts section and find the the most relevant one.
                2.1. If the question is similar to a prompt, extract the corresponding answer from the "response" column of the most similar prompt in the prompts section.
                2.2. If no similar prompt is found, check whether the question is related to the report.
                    2.2.1. If related, extract the answer from the report.
                        2.2.1.1 If no information is available in the report, provide an answer using public references.
                    2.2.2. If the question is unrelated, return "No response found" without answering from public references.

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

        self.rag_prompt = rag_prompt

    def format_docs(self, docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def get_session_history(self, user_id: str, conversation_id: str):
        return SQLChatMessageHistory(
            f"{user_id}--{conversation_id}", "sqlite:///memory.db"
        )

    def initial_qa_chain(self):
        qa_chain = (
            RunnablePassthrough.assign(
                prompts=lambda x: self.format_docs(
                    self.csv_vectorstore.as_retriever(search_type="mmr").invoke(
                        x["question"]
                    )
                )
            )
            | RunnablePassthrough.assign(
                report=lambda x: self.format_docs(
                    self.pdf_vectorstore.as_retriever(search_type="mmr").invoke(
                        x["question"]
                    )
                )
            )
            | self.rag_prompt
            | self._model
            | StrOutputParser()
        )

        with_message_history = RunnableWithMessageHistory(
            qa_chain,
            self.get_session_history,
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

        self.qa_with_message_history = with_message_history

    def get_response(
        self, question: str, user_id: str = "1", conversation_id: str = "1"
    ):
        with get_openai_callback() as cb:
            result = self.qa_with_message_history.invoke(
                {"question": question},
                config={
                    "configurable": {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                    }
                },
            )
            logging.info(cb)

        return f"Assistant > {result}\n" + "-" * 50


assistant = Assistant()
assistant.upload_pdf_files(
    ["files/Sponge Iron report new version 2.pdf", "files/AI and Bio Slides Ver2.pdf"]
)
assistant.upload_csv_files(["files/prompts.csv"])

while True:
    user_message = input("User > ")
    response = assistant.get_response(user_message)
    print(response)
