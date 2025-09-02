from langchain_core.documents import Document
import pandas as pd

def df_to_documents(df: pd.DataFrame, text_column:str, metadata_columns: list[str]=None) -> list[Document]:
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
