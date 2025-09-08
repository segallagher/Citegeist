from langchain_chroma import Chroma
from langchain_core.vectorstores import VectorStoreRetriever
from langchain_core.runnables import RunnableLambda
from langchain_community.retrievers import ArxivRetriever
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

# Create retriever
retriever: VectorStoreRetriever = vectorstore.as_retriever(search_kwargs={"k": 5})
# retriever = ArxivRetriever(
#     top_k_results=5,
#     get_full_documents=True,
# )

question = "What is a good model for semantic segmentation of an image?"
question = "How does GoogLeNet work? Why is it so much faster than other models?"
question = "How does GoogLeNet model work?"
question = "How can we explain how AI models decide what they make?"
# question = "What is the ImageBind model?"
docs = retriever.invoke(question)

def format_docs(docs):
    print(docs[0])
    return "\\n\\n".join(doc.page_content for doc in docs)

# runnable_format_docs = RunnableLambda(format_docs)

chain = Prompts.CITES_SOURCES | llm_model()
response = chain.invoke({"context": docs, "question": question})

print(type(response))
print(response)
with open('response.md','w') as f:
    f.write(response.content)
