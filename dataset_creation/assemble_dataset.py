from dotenv import load_dotenv
import os
import pandas as pd
from pathlib import Path
import json

# load parameters
load_dotenv(override=True)

# PAPER_DIR
# DATASET_PATH

# Initialize dataframe
condensed_df = pd.DataFrame()

# Fill dataframe
# Get each category
for category in [ f.name for f in os.scandir(os.environ.get("PAPER_DIR")) if f.is_dir()]:
    print(f"Getting papers from {category} category")
    category_path = Path(os.environ.get("PAPER_DIR")) / category
    # Get each File in category
    for file_name in os.listdir(category_path):
        if file_name.endswith(".json"):
            file_path = Path(category_path) / file_name

            if not file_path.exists():
                raise FileNotFoundError(f"File does not exist: {file_path}")
            
            with open(file_path, 'r') as f:
                # Clean rows that are null
                data = json.load(f)
                clean_data = [row for row in data if row is not None]

                # Clean individual columns, add category information
                clean_data = [
                        {
                            **row,
                            "abstract":row["abstract"].strip(),
                            "category":category, 
                        }
                        for row in clean_data
                    ]

                current_df = pd.DataFrame(clean_data)
                condensed_df = pd.concat([condensed_df, current_df], axis=0)

# Save to file
os.makedirs(Path(os.environ.get("DATASET_PATH")).parent, exist_ok=True)
dataset_path = Path(os.environ.get("DATASET_PATH"))
condensed_df.to_csv(dataset_path, index=False)
print(f"Dataset created at {dataset_path}")
print(f"Num Rows: {condensed_df.shape[0]}\nNum Cols: {condensed_df.shape[1]}")
