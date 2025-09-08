## To Use Scraper
create `.env` file with values
- `DOWNLOAD_DIR` = your output directory
- `CATEGORY` = your arxiv category to download
- `START_YEAR` = set to the first year you want to download as an integer
- `END_YEAR_INCLUSIVE` = set to the last year you want to download as an integer

## To Condense Downloaded Dataset
create `.env` file with values
- `DOWNLOAD_DIR` = folder with your papers downloaded by year
- `DATASET_DIR` = folder where your condensed dataset will be

## Why create a Dataset rather than use someone elses?
Prior to this work there was a dataset on kaggle called "Arxiv.org AI Research Papers Dataset". It had all the columns needed, but only had 10000 articles and was last updated in 2023. Not only is being 2 years out of date in AI research like being a living dinosaur, 2024 and 2025 contained even larger amounts of knowledge than any year previously. The rate of research paper production is currently on an upwards trend, and as such being up to date is more and more important.