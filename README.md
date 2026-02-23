# Citegeist
Answers your AI related questions with information from research papers using a RAG LLM system

# Requirements
Nvidia GPU \
Docker \
Docker-compose

# Installation
### Install Docker and Docker-compose
https://www.docker.com/get-started/ \
https://docs.docker.com/compose/install/

### Clone project
run command \
`git clone https://github.com/segallagher/Citegeist.git`

### Create vectorstore from papers
1. **Create `.env` file with the following values** \
```env
DATASET_PATH="data/arxiv_dataset/dataset.csv"
VECTORSTORE_DIR="vectorstore"
PAPER_DIR="published/papers"
```
2. **Assemble papers into one csv file by running** \
`python .\dataset_creation\assemble_dataset.py`
3. **Create vectorstore from dataset** \
`python .\create_vectorstore_db.py`

# Use
### Start Project
Enter directory for Citegeist in terminal \
`cd Citegeist` \
run docker-compse \
`docker-compose up`

### Access webui
In your browser go to the webui at \
`http://localhost:80`

### Ask Citegeist an AI research question
In the chatbox, enter your question and get your answers.

# Benchmarking
### Env vars
Add the following to your `.env` file \
```env
EMBED_MODEL_TYPE="ollama"
EMBED_MODEL="mxbai-embed-large:latest"
LLM_MODEL_TYPE="ollama"
LLM_MODEL="llama3.2:3b"
OLLAMA_HOST="http://localhost:11434"
```

### Evaluate
1. Evaluation happens in `evaluate_rag.py`. Depending on which type of evaluation you want to do you must set the `OPERATION` parameter to the type of evaluation you want to do.
2. The questions are stored in `data_analysis/questions.json` \
Each question will be answered by the RAG LLM system and then judged by `llama3.1:8b`
3. Run `evaluate_rag.py` \
This will likely take a while running ollama on your computer


