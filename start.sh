#!/bin/bash

# Load env vars
if [ -f .env ]; then
    echo "Loading environment variables from .env"
    # get env, strip carriage returns
    export $(grep -v '^#' .env | sed 's/\r$//' | xargs)
fi

# Default env var values
: "${DATASET_PATH:=data/arxiv_dataset/dataset.csv}"
: "${VECTORSTORE_DIR:=vectorstore}"

: "${OLLAMA_HOST:=http://ollama:11434}"
: "${EMBED_MODEL_TYPE:=ollama}"
: "${EMBED_MODEL:=mxbai-embed-large:latest}"
: "${LLM_MODEL_TYPE:=ollama}"
: "${LLM_MODEL:=llama3.2:3b}"

: "${PAPER_DIR:=arxiv_papers}"
: "${CATEGORY:=cs.CV,cs.AI,cs.CL}"
: "${START_YEAR:=1993}"
: "${END_YEAR_INCLUSIVE:=2025}"

# Export env vars
export DATASET_PATH
export VECTORSTORE_DIR

export OLLAMA_HOST
export EMBED_MODEL_TYPE
export EMBED_MODEL
export LLM_MODEL_TYPE
export LLM_MODEL

export PAPER_DIR
export CATEGORY
export START_YEAR
export END_YEAR_INCLUSIVE

echo "$DATASET_PATH"
echo "$VECTORSTORE_DIR"
echo "$OLLAMA_HOST"
echo "$EMBED_MODEL_TYPE"
echo "$EMBED_MODEL"
echo "$LLM_MODEL_TYPE"
echo "$LLM_MODEL"
echo "$PAPER_DIR"
echo "$CATEGORY"
echo "$START_YEAR"
echo "$END_YEAR_INCLUSIVE"

echo "Starting Citegeist"

# create venv
python3 -m venv .venv
source .venv/bin/activate
# pip install -r requirements.txt
echo -n "$DATASET_PATH" | xxd >> "tmp.txt"

# If vectorstore not found, create vectorstore
if [ ! -d "$VECTORSTORE_DIR" ]; then
    echo "Not found $VECTORSTORE_DIR"
    # If no dataset try assembling dataset from papers
    if [ ! -f "$DATASET_PATH" ]; then
        echo "Not found $DATASET_PATH"
        # If papers not found, ask user to run scraper
        if [ ! -d "$PAPER_DIR" ]; then
            echo "Not found $PAPER_DIR"
            echo "No papers, cannot create vectorstore"
            echo "Please run the scraper or download paper collection"
            exit 1
        fi
        # Papers found but no dataset, create one from papers
        echo "Assembling dataset"
        python3 dataset_creation/assemble_dataset.py
    fi
    echo "Generating vectorstore"
    python3 create_vectorstore_db.py
fi
