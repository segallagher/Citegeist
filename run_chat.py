from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
import os
from dotenv import load_dotenv

from models import embedding_model, llm_model
from prompts import Prompts

# load parameters
load_dotenv(override=True)

vectorstore: Chroma = Chroma(
    persist_directory=os.environ["VECTORSTORE_DIR"],
    embedding_function=embedding_model(),
)

retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 2})

question = "What is a good model for semantic segmentation of an image?"
docs = retriever.get_relevant_documents(question)

for doc in docs:
    print(doc.metadata["title"])

chain = Prompts.ANSWER_BASED_ON_CONTEXT | llm_model()
response = chain.invoke({"context": docs, "question": question})

print(response)
