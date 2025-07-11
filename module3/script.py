#!/usr/bin/env python3
import requests
import dlt
from dlt.destinations import qdrant
import json
import os


# Q1
print("-" * 30)
print(f"dlt version: {dlt.__version__}")
print()

@dlt.resource
def zoomcamp_data():
    """Load FAQ data from the zoomcamp repository"""
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    for course in documents_raw:
        course_name = course['course']

        for doc in course['documents']:
            doc['course'] = course_name
            yield doc

# q2
print("Question 2: dlt Pipeline")
print("-" * 30)


qdrant_destination = qdrant(
    qd_path="db.qdrant", 
)

print("Creating and running dlt pipeline...")
pipeline = dlt.pipeline(
    pipeline_name="zoomcamp_pipeline",
    destination=qdrant_destination,
    dataset_name="zoomcamp_tagged_data"
)

load_info = pipeline.run(zoomcamp_data())
print("Pipeline completed!")
print()

print("Pipeline trace:")
print(pipeline.last_trace)
print()

trace_str = str(pipeline.last_trace)
if "Normalized data for the following tables:" in trace_str:
    lines = trace_str.split('\n')
    for i, line in enumerate(lines):
        if "Normalized data for the following tables:" in line:
            for j in range(i+1, min(i+10, len(lines))):
                if "zoomcamp_data" in lines[j] and "rows" in lines[j]:
                    print(f"Found row information: {lines[j].strip()}")
                    break
print()

# Q3
print("Question 3: Embeddings")
print("-" * 30)


meta_json_path = "db.qdrant/meta.json"
if os.path.exists(meta_json_path):
    print(f"Found meta.json at: {meta_json_path}")
    with open(meta_json_path, 'r') as f:
        meta_data = json.load(f)
    
    print("Meta.json contents:")
    print(json.dumps(meta_data, indent=2))
    
    if 'embedding_model' in meta_data:
        print(f"\nEmbedding model used: {meta_data['embedding_model']}")
    else:
        print("\nSearching for embedding model information in metadata...")
        def find_embedding_info(obj, path=""):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_path = f"{path}.{key}" if path else key
                    if 'embed' in key.lower() or 'model' in key.lower():
                        print(f"Found potential embedding info at {new_path}: {value}")
                    find_embedding_info(value, new_path)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    find_embedding_info(item, f"{path}[{i}]")
        
        find_embedding_info(meta_data)
else:
    print(f"meta.json not found at {meta_json_path}")
    print("Checking if db.qdrant directory exists...")
    if os.path.exists("db.qdrant"):
        print("db.qdrant directory contents:")
        for item in os.listdir("db.qdrant"):
            print(f"  {item}")
    else:
        print("db.qdrant directory not found")