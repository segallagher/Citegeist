from util import scrape_arxiv_category
import json
from pathlib import Path

output_dir = "ai_papers"
path = Path(output_dir)
for year in range(1993,2025):
    papers = scrape_arxiv_category(years=[year], batch_size=2000, scrape_delay_secs=15)
    path.mkdir(parents=True, exist_ok=True)
    with open(path/ f"{year}_papers.json", "w") as f:
        json.dump(papers, f, indent=4)