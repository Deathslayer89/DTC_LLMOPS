#!/usr/bin/env python3
"""
Search Evaluation System
Evaluates different search methods using various metrics
"""

import requests
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import minsearch
from minsearch import VectorSearch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from rouge import Rouge
from qdrant_client import QdrantClient
from qdrant_client.http import models


def hit_rate(relevance_total):
    """Calculate hit rate metric"""
    cnt = 0
    for line in relevance_total:
        if True in line:
            cnt = cnt + 1
    return cnt / len(relevance_total)


def mrr(relevance_total):
    """Calculate Mean Reciprocal Rank metric"""
    total_score = 0.0
    for line in relevance_total:
        for rank in range(len(line)):
            if line[rank] == True:
                total_score = total_score + 1 / (rank + 1)
                break
    return total_score / len(relevance_total)


def evaluate(ground_truth, search_function):
    """Evaluate search function using ground truth data"""
    relevance_total = []
    
    for q in tqdm(ground_truth, desc="Evaluating search"):
        doc_id = q['document']
        results = search_function(q)
        relevance = [d['id'] == doc_id for d in results]
        relevance_total.append(relevance)
    
    return {
        'hit_rate': hit_rate(relevance_total),
        'mrr': mrr(relevance_total),
    }


def cosine(u, v):
    """Calculate cosine similarity between two vectors"""
    u_norm = np.sqrt(u.dot(u))
    v_norm = np.sqrt(v.dot(v))
    return u.dot(v) / (u_norm * v_norm)


def normalize(u):
    """Normalize a vector"""
    norm = np.sqrt(u.dot(u))
    return u / norm


def main():
    print("Starting Search Evaluation System...")
    
    # Load evaluation data
    print("\nLoading data...")
    url_prefix = 'https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main/03-evaluation/'
    docs_url = url_prefix + 'search_evaluation/documents-with-ids.json'
    documents = requests.get(docs_url).json()
    print(f"Loaded {len(documents)} documents")
    
    ground_truth_url = url_prefix + 'search_evaluation/ground-truth-data.csv'
    df_ground_truth = pd.read_csv(ground_truth_url)
    ground_truth = df_ground_truth.to_dict(orient='records')
    print(f"Loaded {len(ground_truth)} ground truth queries")
    
    
    # Minsearch text evaluation
    print("\n=== Minsearch Text Search ===")
    index = minsearch.Index(
        text_fields=["question", "section", "text"],
        keyword_fields=["course", "id"]
    )
    index.fit(documents)
    
    def search_minsearch(query):
        boost = {'question': 1.5, 'section': 0.1}
        results = index.search(
            query=query['question'],
            filter_dict={'course': query['course']},
            boost_dict=boost,
            num_results=5
        )
        return results
    
    metrics_minsearch = evaluate(ground_truth, search_minsearch)
    hit_rate_minsearch = metrics_minsearch['hit_rate']
    print(f"Hit Rate: {hit_rate_minsearch:.3f}")
    
    # Vector search for question only
    print("\n=== Vector Search (Question Only) ===")
    texts_question = []
    for doc in documents:
        t = doc['question']
        texts_question.append(t)
    
    pipeline_question = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )
    X_question = pipeline_question.fit_transform(texts_question)
    
    vindex_question = VectorSearch(keyword_fields={'course'})
    vindex_question.fit(X_question, documents)
    
    def search_vector_question(query):
        v_query = pipeline_question.transform([query['question']])
        results = vindex_question.search(
            v_query[0],
            filter_dict={'course': query['course']},
            num_results=5
        )
        return results
    
    metrics_question = evaluate(ground_truth, search_vector_question)
    mrr_question = metrics_question['mrr']
    print(f"MRR: {mrr_question:.3f}")
    
    # Vector search for question and answer
    print("\n=== Vector Search (Question + Answer) ===")
    texts_qa = []
    for doc in documents:
        t = doc['question'] + ' ' + doc['text']
        texts_qa.append(t)
    
    pipeline_qa = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )
    X_qa = pipeline_qa.fit_transform(texts_qa)
    
    vindex_qa = VectorSearch(keyword_fields={'course'})
    vindex_qa.fit(X_qa, documents)
    
    def search_vector_qa(query):
        v_query = pipeline_qa.transform([query['question']])
        results = vindex_qa.search(
            v_query[0],
            filter_dict={'course': query['course']},
            num_results=5
        )
        return results
    
    metrics_qa = evaluate(ground_truth, search_vector_qa)
    hit_rate_qa = metrics_qa['hit_rate']
    print(f"Hit Rate: {hit_rate_qa:.3f}")
    
    # Qdrant evaluation
    print("\n=== Qdrant Vector Search ===")
    try:
        from qdrant_evaluation import run_qdrant_evaluation
        qdrant_results = run_qdrant_evaluation(documents, ground_truth)
        print(f"MRR: {qdrant_results['qdrant_mrr']:.3f}")
        print(f"Hit Rate: {qdrant_results['qdrant_hit_rate']:.3f}")
    except Exception as e:
        print(f"Qdrant evaluation failed: {e}")
        print("Using fallback results")
    
    # Cosine similarity evaluation
    print("\n=== Cosine Similarity Analysis ===")
    results_url = url_prefix + 'rag_evaluation/data/results-gpt4o-mini.csv'
    df_results = pd.read_csv(results_url)
    print(f"Loaded {len(df_results)} result pairs")
    
    # Create pipeline for embeddings
    pipeline_cosine = make_pipeline(
        TfidfVectorizer(min_df=3),
        TruncatedSVD(n_components=128, random_state=1)
    )
    
    # Fit on all text data
    all_texts = (df_results.answer_llm + ' ' + 
                df_results.answer_orig + ' ' + 
                df_results.question)
    pipeline_cosine.fit(all_texts)
    
    # Calculate cosine similarities
    cosine_similarities = []
    for idx, row in df_results.iterrows():
        v_llm = pipeline_cosine.transform([row.answer_llm])[0]
        v_orig = pipeline_cosine.transform([row.answer_orig])[0]
        cos_sim = cosine(v_llm, v_orig)
        cosine_similarities.append(cos_sim)
    
    avg_cosine = np.mean(cosine_similarities)
    print(f"Average Cosine Similarity: {avg_cosine:.3f}")
    
    # ROUGE evaluation
    print("\n=== ROUGE Evaluation ===")
    rouge_scorer = Rouge()
    
    # Calculate for entire dataframe
    rouge_f1_scores = []
    for idx, row in tqdm(df_results.iterrows(), total=len(df_results), desc="Calculating ROUGE"):
        try:
            scores = rouge_scorer.get_scores(row.answer_llm, row.answer_orig)[0]
            rouge_f1_scores.append(scores['rouge-1']['f'])
        except:
            # Handle edge cases where ROUGE fails
            rouge_f1_scores.append(0.0)
    
    avg_rouge_f1 = np.mean(rouge_f1_scores)
    print(f"Average ROUGE-1 F1: {avg_rouge_f1:.3f}")
    
   
    


if __name__ == "__main__":
    main()