from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.language_models.chat_models import BaseChatModel

def retrieve_context(message:str, vectorstore:Chroma,  n_docs:int=5):
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": n_docs})
    docs = retriever.invoke(message)
    return docs
