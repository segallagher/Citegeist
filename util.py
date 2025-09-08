from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup
import re
from datetime import datetime
from typing import Any
import time
import random

from models import embedding_model

def scrape_delay(delay: int=None, delay_lower_bound: int=None, delay_upper_bound:int=None):
    """
    Waits some time to avoid ddos-ing host
    If delay_lower_bound and delay_upper_bound set, delay a random time between those values
    Otherwise delay for delay seconds
    """
    if delay_lower_bound and delay_upper_bound:
        delay = random.uniform(delay_lower_bound, delay_upper_bound)
    time.sleep(delay)

def get_arxiv_paper_metadata(url:str, scrape_delay_secs:int=15) -> dict[str, Any]:
    """
    Gets the metadata of a single paper
    """
    response = requests.get(url)
    if response.status_code != 200:
        print(f"Failed to fetch {url}")
        return None
    
    soup = BeautifulSoup(response.text, 'html.parser')

    metadata = {"url": url}

    # Title
    title_tag = soup.find('h1', class_='title mathjax')
    metadata["title"] = str(title_tag.find_all(string=True, recursive=False)[0])

    # Authors
    authors_tag = soup.find('div', class_='authors')
    metadata["authors"] = [tag.text for tag in authors_tag.find_all('a')]

    # Abstract
    abstract_tag = soup.find('blockquote', class_='abstract mathjax')
    metadata["abstract"] = str(abstract_tag.find_all(string=True, recursive=False)[1])

    # Submission Date
    submission_tag = soup.find('div', class_='dateline')
    date_match = re.findall(r'\b\d{1,2} \w{3} \d{4}\b', submission_tag.text)
    dates = [datetime.strptime(ds, "%d %b %Y") for ds in date_match]
    metadata["submission_date"] = dates[0].isoformat()

    # Last Modified Date
    if len(dates) > 1:
        metadata["last_modified_date"] = dates[1].isoformat()
    else:
        metadata["last_modified_date"] = dates[0].isoformat()

    # Could include Category information

    scrape_delay(scrape_delay_secs)
    return metadata
    

def parse_arxiv_list(html:str, scrape_delay_secs:int=15, progress_bar=None) -> dict[str, dict]:
    """
    Gets the metadata of all papers of an arxiv category search
    """
    soup = BeautifulSoup(html, "html.parser")
    parsed_page = {}
    entries_tag = soup.find('div', class_='paging')
    parsed_page["total_entries"] = int(re.search(r"[\d]+", entries_tag.text).group())

    papers = []
    dt_tags = soup.find_all('dt')
    for dt in dt_tags:
        link_tag = dt.find('a', title="Abstract")
        if link_tag and link_tag['href'].startswith('/abs/'):
            metadata = get_arxiv_paper_metadata(f"https://arxiv.org{link_tag['href']}", scrape_delay_secs=scrape_delay_secs)
            papers.append(metadata)
            # update progress_bar
            if progress_bar:
                progress_bar.update(1)
    parsed_page["papers"] = papers
    
    scrape_delay(scrape_delay_secs)
    return parsed_page

def scrape_arxiv_category(
        category:str="cs.AI",
        start_year:int=None,
        end_year:int=None,
        years:list=None,
        batch_size:int=2000,
        scrape_delay_secs:int=15,
    ) -> list[dict]:
    """
    Gets the paper metadata from all papers for a given category and given years
    """
    # Validate inputs
    valid_batch_sizes = [25, 50, 100, 250, 500, 1000, 2000]
    if batch_size not in valid_batch_sizes:
        raise ValueError(f"batch_size must be one of {valid_batch_sizes}")
    

    # Generate range of years
    if start_year and end_year:
        years = range(start_year, end_year)
    elif start_year and not end_year:
        years = range(start_year, datetime.now().year)
    
    papers = []
    for year in years:
        # Get papers and append them to papers list
        url = f"https://arxiv.org/list/{category}/{year}?skip=0&show={batch_size}"
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return None
        parsed = parse_arxiv_list(response.text, scrape_delay_secs=scrape_delay_secs)
        papers.extend(parsed["papers"])
        total_entries = parsed["total_entries"]
        # Repeat for each subset of searches
        with tqdm(total=total_entries, desc=f"Processing {year}", initial=batch_size) as pbar:
            for i in range(batch_size, total_entries, batch_size):
                url = f"https://arxiv.org/list/{category}/{year}?skip={i}&show={batch_size}"
                response = requests.get(url)
                # Check reponse, if fail try next link
                if response.status_code != 200:
                    print(f"Failed to fetch {url}")
                    continue
                parsed = parse_arxiv_list(response.text, progress_bar=pbar, scrape_delay_secs=scrape_delay_secs)
                papers.extend(parsed["papers"])

    scrape_delay(scrape_delay_secs)
    return papers



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

    print(f"Dataset at {dataset_path} loaded")
    print(f"Datset shape: {dataset.shape}")

    # Convert dataset to documents
    documents = df_to_documents(dataset, text_column=primary_column)

    # Create vectorstore
    db = Chroma(
        embedding_function=embedding_model(),
        persist_directory=str(persist_directory),
    )

    pbar = tqdm(total=len(documents), desc="Inserting documents")
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]
        db.add_documents(batch)
        pbar.update(batch_size)
