from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import shutil
from pathlib import Path
from dotenv import load_dotenv
import os

# load parameters
load_dotenv(override=True)
data_path = Path(os.environ["DATA_DIR"])
dataset_name = os.environ["DATASET_FILE"]

# initialize dataset
dataset: pd.DataFrame = pd.DataFrame(columns=[
    "authors",
    "journal_entry",
    "pdf_url",
    "publish_date",
    "title",
    "abstract",
    ])

### JUSTIFICATION
## We want to get metadata on arxiv research articles
## we cannot use the existing "Arxiv.org AI Research Papers Dataset" dataset since it is out of date
## We cannot skip this step by using LangChains ArxivRetriever since it does not provide highly relevant data

# initialize kaggle api
api = KaggleApi()
api.authenticate()

# download dataset "Arxiv.org AI Research Papers Dataset"
api.dataset_download_files(
    dataset="yasirabdaali/arxivorg-ai-research-papers-dataset",
    path=data_path / "data",
    unzip=True,
)

# load data
data: pd.DataFrame = pd.read_csv(data_path / "data" / "arxiv_ai.csv")
# delete downloaded file
shutil.rmtree(data_path / "data")

# format data
data.drop(columns=["categories", "comment", "doi", "journal_ref", "primary_category", "updated"], inplace=True)
data.rename(
        columns={
            "entry_id": "journal_entry",
            "published": "publish_date",
            "summary": "abstract",
        },
        inplace=True
    )

# add data to dataset
dataset = pd.concat([dataset, data], axis=0, ignore_index=True)

# write to file
dataset.to_csv(data_path / dataset_name)
