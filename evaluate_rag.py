from langchain_chroma import Chroma
import os
import json
from pathlib import Path
from dotenv import load_dotenv
import pandas as pd

from app.rag.models import embedding_model
from data_analysis.eval_util import evaluate_retrieval, evaluate_retrieval_llm, evaluate_response_llm, comparitive_llm_judge

# Load env vars
load_dotenv()

# Constants
# Valid operations [score_independent_with_context, score_independent_no_context, compare_responses, score_context_relevance]
OPERATION = "score_independent_no_context"
SAVE_SCORES = True
SAVE_CSV = True
SAVE_HTML = True
OUTPUT_DIR = Path("outputs")

# Load vectorstore and set of questions
vectorstore: Chroma = Chroma(
    persist_directory=os.environ["VECTORSTORE_DIR"],
    embedding_function=embedding_model(),
)
with open(Path('data_analysis') /'questions.json', 'r') as f:
    questions = json.load(f)


if OPERATION == "score_independent_with_context":
    # Grades LLM responses with and without RAG independently with LLM-as-a-judge
    # Judge LLM uses the DOES_RESPONSE_ANSWER_QUESTION prompt
    # Prints the average performance with the model
    # Runs LLM with RAG context

    context_df: pd.DataFrame = evaluate_response_llm(questions, vectorstore=vectorstore, n_docs=5)
    context_print_str = f"Does reponse With-Context answer well:\n{context_df['response_answers_question'].value_counts(normalize=True)}\n"

    print(context_print_str)
    if SAVE_CSV:
        context_df.to_csv(OUTPUT_DIR / "response_with_context.csv", index=False)
    if SAVE_HTML:
        context_df.to_html(OUTPUT_DIR / "response_with_context.html", index=False)
    if SAVE_SCORES:
        with open(OUTPUT_DIR / "response_scores.txt", 'w') as f:
            f.write(context_print_str)

elif OPERATION == "score_independent_no_context":
    # Grades LLM responses with and without RAG independently with LLM-as-a-judge
    # Judge LLM uses the DOES_RESPONSE_ANSWER_QUESTION prompt
    # Prints the average performance with the model
    # Runs LLM without RAG context

    no_context_df: pd.DataFrame = evaluate_response_llm(questions, vectorstore=vectorstore, with_context=False)
    no_context_print_str = f"Does reponse No-Context answer well:\n{no_context_df['response_answers_question'].value_counts(normalize=True)}\n"

    print(no_context_print_str)
    if SAVE_CSV:
        no_context_df.to_csv(OUTPUT_DIR / "response_no_context.csv", index=False)
    if SAVE_HTML:
        no_context_df.to_html(OUTPUT_DIR / "response_no_context.html", index=False)
    if SAVE_SCORES:
        with open(OUTPUT_DIR / "response_scores.txt", 'w') as f:
            f.write(no_context_print_str)

elif OPERATION == "compare_responses":
    # Compares responses with and without RAG context with LLM-as-a-judge
    # A judge LLM determines which response it prefers based off the CHOOSE_BETTER_RESPONSE prompt
    # Prints the rate at which the judge preferred each response
    # Also prints the proportion of time spent on LLM generation, only in terminal
    # Judge LLM is llama3.1:8b
    comparison_df = comparitive_llm_judge(questions, vectorstore=vectorstore, n_docs=5)

    comparison_scores_print_str = f"Which response is more satisfactory:\n{comparison_df['preferred_answer'].value_counts(normalize=True)}\n"
    print(comparison_scores_print_str)
    if SAVE_CSV:
        comparison_df.to_csv(OUTPUT_DIR / 'comparison.csv', index=False)
    if SAVE_HTML:
        comparison_df.to_html(OUTPUT_DIR / 'comparison.html', index=False)
    if SAVE_SCORES:
        with open(OUTPUT_DIR / "comparison_scores.txt", 'w') as f:
            f.write(comparison_scores_print_str)

elif OPERATION == "score_context_relevance":
    # Uses LLM-as-a-judge
    context_scores_df = evaluate_retrieval_llm(questions, vectorstore)
    context_scores_print_str = f"Avg Relevance Score: {context_scores_df['relevance'].mean()}"

    print(context_scores_print_str)
    if SAVE_CSV:
        context_scores_df.to_csv(OUTPUT_DIR / 'context_relevance_score.csv', index=False)
    if SAVE_HTML:
        context_scores_df.to_html(OUTPUT_DIR / 'context_relevance_score.html', index=False)
    if SAVE_SCORES:
        with open(OUTPUT_DIR / "context_relevance_scores.txt", 'w') as f:
            f.write(context_scores_print_str)

elif OPERATION == "ground_truth_compare":
    # Compare llm performance against ground truth labeled data
    # DO NOT USE THIS ONE UNLESS YOU HAVE SUFFICIENT HUMAN CREATED DATA
    # uses ground_truth question pairs for comparison, not fully fleshed out
    with open(Path('data_analysis') /'question_ground_truth.json', 'r') as f:
        question_ground_truth_pairs = json.load(f)
        
    print("Ground truth comparison not implemented")

else:
    print(f"Invalid operation: {OPERATION}")
