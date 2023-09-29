import json
import os
from pathlib import Path
from typing import Dict, List

import openai
from langchain import OpenAI
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.docstore.document import Document
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.vectorstores import DeepLake


def create_web_search_tool(
    serper_api_key: str, tool_name: str, tool_description: str
) -> Tool:
    web_search = GoogleSerperAPIWrapper(
        serper_api_key=serper_api_key,
    )
    web_search_tool = Tool(
        name=tool_name,
        func=web_search.run,
        description=tool_description,
    )
    return web_search_tool


def create_vectorstore_search_tool(
    model_dir: str, tool_name: str, tool_description: str
) -> Tool:
    retrieval_llm = OpenAI(temperature=0)
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    vectorstore_retriever = RetrievalQA.from_chain_type(
        llm=retrieval_llm, retriever=db.as_retriever(), chain_type="stuff"
    )
    vectorstore_search_tool = Tool(
        name=tool_name,
        func=vectorstore_retriever.run,
        description=tool_description,
    )
    return vectorstore_search_tool


def build_qa_chain(retriever, qa_chain_prompt):
    llm = ChatOpenAI()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm, retriever=retriever, chain_type_kwargs={"prompt": qa_chain_prompt}
    )
    return qa_chain


def build_qa_prompt():
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Use three sentences maximum and keep the answer as concise as possible. 
    {context}
    Question: {question}
    Helpful Answer: Lets take it step by step"""
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)
    return QA_CHAIN_PROMPT


def json_to_doc(dataset_path):
    data: List[Dict] = json.loads(Path(dataset_path).read_text())
    docs = [
        Document(page_content=d["page_content"], metadata=d["metadata"]) for d in data
    ]
    return docs


def get_token_count(text, model_name="text-davinci-003"):
    tokenizer_model = openai.model(model_name)
    return tokenizer_model.token_count(text)


def set_openai_api_key(api_key):
    os.environ["OPENAI_API_KEY"] = api_key


def build_retriever(model_dir):
    db = DeepLake(
        dataset_path=model_dir,
        embedding=OpenAIEmbeddings(),
        read_only=True,
    )
    retriever = db.as_retriever(
        search_type="mmr",  # Also test "similarity"
        search_kwargs={"k": 3},
    )
    return retriever
