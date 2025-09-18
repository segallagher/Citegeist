from langchain_chroma import Chroma
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from app.rag.models import embedding_model
from data_analysis.eval_util import evaluate_retrieval, evaluate_retrieval_llm, evaluate_response_llm, comparitive_llm_judge

load_dotenv()

# with open(Path('data_analysis') /'question_ground_truth.json', 'r') as f:
#     question_ground_truth_pairs = json.load(f)
with open(Path('data_analysis') /'questions.json', 'r') as f:
    questions = json.load(f)
vectorstore: Chroma = Chroma(
    persist_directory=os.environ["VECTORSTORE_DIR"],
    embedding_function=embedding_model(),
)

# score_df = evaluate_retrieval_llm(questions, vectorstore, save_dir="context_score.csv")
# print(f"Avg Relevance Score: {score_df['relevance'].mean()}")

# context_df = evaluate_response_llm(questions, vectorstore=vectorstore, n_docs=5, save_dir="context.csv")
# print(f"Avg Relevance Score: {context_df['answers_question'].mean()}")

# no_context_df = evaluate_response_llm(questions, save_dir="no_context.csv")
# print(f"Avg No Context Relevance Score: {no_context_df['answers_question'].mean()}")

judge_df = comparitive_llm_judge(questions, vectorstore=vectorstore, n_docs=5)
judge_df.to_csv('judge.csv', index=False)
judge_df.to_html('judge.html', index=False)

print(judge_df['prefered_answer'].value_counts(normalize=True))