from dotenv import load_dotenv
from typing import Any, Dict, List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain_cohere import CohereEmbeddings
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_community.document_loaders import RecursiveUrlLoader
load_dotenv()

def url(url_link):
    loader = RecursiveUrlLoader(url_link)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(data)


    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")

    vector_db = PineconeVectorStore.from_documents(texts, embeddings,index_name = "test3")

    return vector_db

# PINECONE_INDEX_NAME= 'test3'

def run_llm(query: str, chat_history: List[Dict[str, Any]] = []):
    embeddings = CohereEmbeddings(model="embed-english-light-v3.0")
    PINECONE_INDEX_NAME= 'test3'
    docsearch = PineconeVectorStore(index_name=PINECONE_INDEX_NAME,embedding=embeddings)
    chat = ChatGroq(verbose=True, temperature=0)

    rephrase_prompt = hub.pull("langchain-ai/chat-langchain-rephrase")

    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    stuff_documents_chain = create_stuff_documents_chain(chat, retrieval_qa_chat_prompt)

    history_aware_retriever = create_history_aware_retriever(
        llm=chat, retriever=docsearch.as_retriever(), prompt=rephrase_prompt
    )
    qa = create_retrieval_chain(
        retriever=history_aware_retriever, combine_docs_chain=stuff_documents_chain
    )

    result = qa.invoke(input={"input": query, "chat_history": chat_history})
    return result
