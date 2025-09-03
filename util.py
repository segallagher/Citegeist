from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
from tqdm import tqdm

from models import embedding_model

def df_to_documents(df: pd.DataFrame, text_column:str, metadata_columns: list[str]=None) -> list[Document]:
    """
    Converts a pandas dataframe to documents

    Args:
        df (pd.DataFrame): Pandas DataFrame
        text_column (str): Name of primary text column
        metadata_columns (list[str], optional): List of column names to include as metadata.
            If None, all columns except text_column will be used.

    Returns:
        list[Document]: A list of LangChain Document objects.

    Raises:
        ValueError: If text_column or specified metadata column not in the DataFrame
    """

    # get metadata_columns
    if metadata_columns is None:
        metadata_columns = [col for col in df.columns if col != text_column]
    
    # Format documents
    docs = [
        Document(
            page_content=row[text_column],
            metadata={col: row[col] for col in metadata_columns}
        )
        for _, row in df.iterrows()
    ]
    return docs

def create_vectorstore(dataset_path:str, persist_directory:str, primary_column:str="abstract", batch_size:int=100) -> None:
    """
    Creates a persistant Croma vector store from a dataset.

    Args:
        dataset_path (str): Path to dataset CSV
        persist_directory (str): Path where vectorstore persists
        primary_column (str, optional): primary text column for embedding, defaults to "abstract"
        batch_size (int, optional): batch size for adding documents to vectorstore
    
    Returns:
        None
    
    Raises:
        ValueError: If primary_column not in dataset
    """

    # load dataset into documents
    dataset = pd.read_csv(dataset_path)

    # Convert dataset to documents
    documents = df_to_documents(dataset, text_column=primary_column)

    # Create vectorstore
    db = Chroma(
        embedding_function=embedding_model(),
        persist_directory=str(persist_directory),
    )

    for i in tqdm(range(0, len(documents), batch_size), desc="Inserting documents"):
        batch = documents[i : i + batch_size]
        db.add_documents(batch)

