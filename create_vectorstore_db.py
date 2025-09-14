import os
from pathlib import Path

from dataset_creation.vectorstore_utils import create_vectorstore

# Requires env vars
# DATASET_PATH
# VECTORSTORE_DIR

create_vectorstore(
    dataset_path=Path(os.environ["DATASET_PATH"]),
    persist_directory=os.environ["VECTORSTORE_DIR"],
    primary_column="abstract",
    batch_size=64,
)
