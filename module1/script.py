#!/usr/bin/env python3

import requests
import json
import tiktoken


# Base URL for Elasticsearch
ES_URL = "http://localhost:9200"

# Get the data
print("\nGetting FAQ data...")
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []
for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

print(f"Loaded {len(documents)} documents")

# Q2: Index the data
print("\nQ2: Indexing data...")

# Delete index if exists
try:
    requests.delete(f"{ES_URL}/course-questions")
except:
    pass

# Create index with proper mapping
index_settings = {
    "mappings": {
        "properties": {
            "course": {"type": "keyword"},
            "question": {"type": "text"},
            "text": {"type": "text"}
        }
    }
}

response = requests.put(f"{ES_URL}/course-questions", 
                       headers={"Content-Type": "application/json"},
                       data=json.dumps(index_settings))
print(f"Index creation response: {response.status_code}")

# Index documents
for i, doc in enumerate(documents):
    response = requests.post(f"{ES_URL}/course-questions/_doc", 
                           headers={"Content-Type": "application/json"},
                           data=json.dumps(doc))
    if i % 100 == 0:
        print(f"Indexed {i} documents...")

print("Q2: The function used for adding data to Elasticsearch is 'index'")
print("Answer: index")

# Q3: Search with boosting
print("\nQ3: Searching with boosting...")

query = {
    "size": 5,
    "query": {
        "multi_match": {
            "query": "How do execute a command on a Kubernetes pod?",
            "fields": ["question^4", "text"],
            "type": "best_fields"
        }
    }
}

response = requests.post(f"{ES_URL}/course-questions/_search", 
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(query))
search_results = response.json()
top_score = search_results['hits']['hits'][0]['_score']
print(f"Q3: Top score: {top_score}")

# Q4: Filtering by course
print("\nQ4: Filtering by course...")

query_filtered = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": "How do copy a file to a Docker container?",
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
            },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                }
            }
        }
    }
}

response = requests.post(f"{ES_URL}/course-questions/_search", 
                        headers={"Content-Type": "application/json"},
                        data=json.dumps(query_filtered))
search_results_filtered = response.json()
third_question = search_results_filtered['hits']['hits'][2]['_source']['question']
print(f"Q4: Third question: {third_question}")

# Q5: Building a prompt
print("\nQ5: Building a prompt...")

context_template = """
Q: {question}
A: {text}
""".strip()

context_entries = []
for hit in search_results_filtered['hits']['hits']:
    doc = hit['_source']
    context_entries.append(context_template.format(question=doc['question'], text=doc['text']))

context = "\n\n".join(context_entries)

prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT:
{context}
""".strip()

prompt = prompt_template.format(
    question="How do copy a file to a Docker container?",
    context=context
)

prompt_length = len(prompt)
print(f"Q5: Prompt length: {prompt_length}")

# Q6: Token counting
print("\nQ6: Counting tokens...")

encoding = tiktoken.encoding_for_model("gpt-4o")
tokens = encoding.encode(prompt)
token_count = len(tokens)
print(f"Q6: Token count: {token_count}")

print("\n" + "="*50)
print("HOMEWORK ANSWERS SUMMARY:")
print("="*50)
print(f"Q1: Build hash: dbcbbbd0bc4924cfeb28929dc05d82d662c527b7")
print(f"Q2: Function used: index")
print(f"Q3: Top score: {top_score}")
print(f"Q4: Third question: {third_question}")
print(f"Q5: Prompt length: {prompt_length}")
print(f"Q6: Token count: {token_count}") 