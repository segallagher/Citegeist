from dotenv import load_dotenv
import os
from pathlib import Path

from util import create_vectorstore


# load parameters
load_dotenv(override=True)

create_vectorstore(
    dataset_path=Path(os.environ["DATA_DIR"]) / os.environ["DATASET_FILE"],
    persist_directory=os.environ["VECTORSTORE_DIR"],
    primary_column="abstract",
    batch_size=64,
)
