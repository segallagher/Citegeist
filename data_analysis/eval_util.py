from langchain_chroma import Chroma
import pandas as pd
import string
from tqdm import tqdm
from datetime import datetime, timedelta

from app.rag.pipeline import retrieve_context
from app.rag.models import llm_model
from app.rag.prompts import Prompts


def evaluate_retrieval(question_ground_truth_pairs:list[dict], vectorstore:Chroma, n_docs:int=5):
    hits = 0
    for pair in question_ground_truth_pairs:
        q = pair["question"]
        gt_id = pair["ground_truth_id"]

        retrieved = retrieve_context(q, vectorstore, n_docs)
        retrieved_ids = [doc.metadata['url'] for doc in retrieved]
        print(q)
        print(retrieved_ids)
        if gt_id in retrieved_ids:
            hits += 1
    recall_at_k = hits / len(question_ground_truth_pairs)
    return recall_at_k


def evaluate_retrieval_llm(questions: list[str],
                           vectorstore:Chroma,
                           n_docs:int=5,
                        ) -> pd.DataFrame:
    # Setup dataframe
    schema = {
        'question': str,
        'context': str, 
        'relevance': int,
        'relevance_scores': 'object',
    }
    score_df: pd.DataFrame = pd.DataFrame(columns=schema).astype(schema)

    # Score each question's context relevance
    pbar = tqdm(total=len(questions) * n_docs, desc="Rating Context")
    for question in questions:
        # Retrieve context for question
        retrieved_context = retrieve_context(question, vectorstore, n_docs)
        # Ask LLM if each retrieved context is relevant
        relevance_scores: list[int] = []
        for doc in retrieved_context:
            chain = Prompts.IS_DOCUMENT_RELEVANT | llm_model()
            response = chain.invoke({"context": doc, "question": question})
            # Lowercase and remove punctuation
            normalized_content:str = ''.join([char for char in response.content.lower() if char not in string.punctuation])
            # convert to integer and Append to relevance_scores
            if normalized_content == "relevant":
                relevance_scores.append(1)
            elif normalized_content == "irrelevant":
                relevance_scores.append(0)
            else:
                print(f"Invalid llm output: {normalized_content}")
            pbar.update(1)

        # Add question and context relevance to dataframe 
        score_df = pd.concat([
                    pd.DataFrame([[
                            question,
                            [doc.page_content for doc in retrieved_context],
                            sum(relevance_scores) / len(relevance_scores),
                            relevance_scores,
                        ]],
                        columns=score_df.columns
                    ),
                    score_df
                ],
            ignore_index=True
        )
    
    # https://www.evidentlyai.com/llm-guide/rag-evaluation
    return score_df


def evaluate_response_llm(questions: list[str],
                          vectorstore:Chroma=None,
                          n_docs:int=5,
                        ) -> pd.DataFrame:
    # Setup dataframe
    schema = {
        'question': str,
        'response': str,
        'response_answers_question': str,
        'reason': str,
    }
    response_df: pd.DataFrame = pd.DataFrame(columns=schema).astype(schema)

    pbar = tqdm(total=len(questions), desc="Rating Context")
    for question in questions:
        # Invoke with or without context
        if n_docs > 0 and vectorstore:
            # Retrieve context
            docs = retrieve_context(question, vectorstore, n_docs)
            # Invoke with context
            chain = Prompts.CITES_SOURCES | llm_model()
            response = chain.invoke({"context": docs, "question": question})
        else:
            # Invoke without context
            chain = Prompts.ANSWER_QUESTION_NO_CONTEXT | llm_model()
            response = chain.invoke({"question": question})

        # Grade responses
        chain = Prompts.DOES_RESPONSE_ANSWER_QUESTION | llm_model()
        judge_response = chain.invoke({"response": response, "question": question})

        does_response_answer_question, reason = [part.strip() for part in judge_response.content.split('|', 1)]

        response_df = pd.concat([
                    pd.DataFrame([[
                            question,
                            response.content,
                            does_response_answer_question,
                            reason,
                        ]],
                        columns=response_df.columns
                    ),
                    response_df
                ],
            ignore_index=True
        )
        pbar.update(1)

    return response_df


def comparitive_llm_judge(questions: list[str],
                          vectorstore:Chroma=None,
                          n_docs:int=5,
                         ) -> pd.DataFrame:
    # Initialize values
    schema = {
        'question': str,
        'response_no_context': str,
        'response_with_context': str,
        'preferred_answer': str,
        'reason': str,
    }
    judge_df: pd.DataFrame = pd.DataFrame(columns=schema).astype(schema)

    time_no_context = timedelta()
    time_with_context = timedelta()
    time_judge = timedelta()

    # Ask the same question twice and see which answer is preferred
    pbar = tqdm(total=len(questions), desc="Rating Context")
    for question in questions:
        # Get Response with retrieval
        # Retrieve context
        start_time = datetime.now()
        docs = retrieve_context(question, vectorstore, n_docs)
        # Invoke with context
        chain = Prompts.CITES_SOURCES | llm_model()
        response_with_context = chain.invoke({"context": docs, "question": question})
        end_time = datetime.now()
        time_with_context += end_time - start_time
        
        # Get Response without retrieval
        start_time = datetime.now()
        chain = Prompts.ANSWER_QUESTION_NO_CONTEXT | llm_model()
        response_no_context = chain.invoke({"question": question})
        end_time = datetime.now()
        time_no_context += end_time - start_time

        # Choose better response
        start_time = datetime.now()
        chain = Prompts.CHOOSE_BETTER_RESPONSE | llm_model(model="llama3.1:8b")
        judge_response = chain.invoke({"answer1": response_no_context, "answer2": response_with_context, "question": question})
        end_time = datetime.now()
        time_judge += end_time - start_time

        preferred_answer, reason = [part.strip() for part in judge_response.content.split('|', 1)]

        # update judge_df
        judge_df = pd.concat([
                    pd.DataFrame([[
                            question,
                            response_no_context.content,
                            response_with_context.content,
                            "no_context" if '1' in preferred_answer.lower() else "context" if '2' in preferred_answer.lower() else "unknown",
                            reason,
                        ]],
                        columns=judge_df.columns
                    ),
                    judge_df
                ],
            ignore_index=True
        )
        pbar.update(1)

    print(f"Total LLM Time: {time_no_context + time_with_context + time_judge}")
    print(f"Time No Context: {time_no_context}")
    print(f"Time With Context: {time_with_context}")
    print(f"Time Judging: {time_judge}")
    return judge_df
