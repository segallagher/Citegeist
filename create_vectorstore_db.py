from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
import os
import pandas
from pathlib import Path

from util import df_to_documents

# load parameters
load_dotenv(override=True)
embd = OllamaEmbeddings(model=os.environ["EMBED_MODEL"])

# load dataset into documents
dataset = pandas.read_csv(Path("./") / os.environ["DATA_PATH"] / os.environ["DATASET_FILE"])

documents = df_to_documents(dataset, text_column="abstract")

vectorstore = Chroma.from_documents(
    documents=documents,
    embedding = embd,
    persist_directory=Path("./") / os.environ["VECTORSTORE_DIR"]
)

