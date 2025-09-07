from scraper import scrape_arxiv_category
import json
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv(override=True)

path = Path(os.environ.get("OUTPUT_DIR"))
for year in range(int(os.environ.get("START_YEAR")),int(os.environ.get("END_YEAR_INCLUSIVE"))+1):
    papers = scrape_arxiv_category(years=[year], batch_size=2000, scrape_delay_secs=15)
    path.mkdir(parents=True, exist_ok=True)
    with open(path/ f"{year}_papers.json", "w") as f:
        json.dump(papers, f, indent=4)