from scraper import scrape_arxiv_category
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(override=True)

# Requiers the following ENV vars
# PAPER_DIR
# CATEGORY
# START_YEAR
# END_YEAR_INCLUSIVE

# Path to dataset_dir/arxiv_category
path = Path(os.environ.get("PAPER_DIR")) / os.environ.get("CATEGORY")

# Scrape each year individually
for year in range(int(os.environ.get("START_YEAR")),int(os.environ.get("END_YEAR_INCLUSIVE"))+1):
    # Get papers
    papers = scrape_arxiv_category(years=[year], batch_size=2000, scrape_delay_secs=0.5, category=os.environ.get("CATEGORY"))

    # Output year to file
    path.mkdir(parents=True, exist_ok=True)
    with open(path/ f"{year}_papers.json", "w") as f:
        json.dump(papers, f, indent=4)