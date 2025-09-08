from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import json

# load parameters
load_dotenv(override=True)

condensed_df = pd.DataFrame()
for file_name in os.listdir(os.environ.get("DOWNLOAD_DIR")):
    if file_name.endswith(".json"):
        file_path = Path(os.environ.get("DOWNLOAD_DIR")) / file_name

        if not file_path.exists():
            raise FileNotFoundError(f"File does not exist: {file_path}")
        
        with open(file_path, 'r') as f:
            # Clean rows that are null
            data = json.load(f)
            clean_data = [row for row in data if row is not None]
            current_df = pd.DataFrame(clean_data)
            condensed_df = pd.concat([condensed_df, current_df], axis=0)

dataset_path = Path(os.environ.get("DATASET_DIR")) / "dataset.csv"
condensed_df.to_csv(dataset_path)
print(f"Dataset created at {dataset_path}")
print(f"Num Rows: {condensed_df.shape[0]}\nNum Cols: {condensed_df.shape[1]}")
