from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.output_parsers import StrOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from .prompts import Prompts
from .models import llm_model
from .util import get_unique_union

def retrieve_context(message:str, vectorstore:Chroma, n_prompts:int=4 ,n_docs:int=2):
    ## Multiquery retrieval
    # Chain for generating multiple queries
    generate_queries = (
        Prompts.MULTI_QUERY
        | llm_model(temperature=0)
        | StrOutputParser()
        | (lambda x: [line for line in x.split('\n') if line.strip()])
    )
    # Initialize retriever
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": n_docs})
    retrieval_chain = generate_queries | retriever.map() | get_unique_union

    docs = retrieval_chain.invoke({"n_prompts": n_prompts, "question": message})
    return docs


def retrieve_context_base(message:str, vectorstore:Chroma,  n_docs:int=5):
    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": n_docs})
    docs = retriever.invoke(message)
    return docs
