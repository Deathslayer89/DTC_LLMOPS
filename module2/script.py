import numpy as np
import requests
from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import VectorParams, Distance


def q1_get_query_embedding():
    """Q1 – embed the query and report min value."""
    query = "I just discovered the course. Can I join now?"
    model_name = "jinaai/jina-embeddings-v2-small-en"
    embedder = TextEmbedding(model_name=model_name)
    embedding = next(embedder.embed([query]))
    print("Q1 – min value in embedding:", round(np.min(embedding), 2))
    return embedding


def q2_similarity(query_vec):
    """Q2 – cosine similarity between query and doc embedding."""
    doc_text = "Can I still join the course after the start date?"
    model_name = "jinaai/jina-embeddings-v2-small-en"
    embedder = TextEmbedding(model_name=model_name)
    doc_vec = next(embedder.embed([doc_text]))
    sim = float(np.dot(query_vec, doc_vec))
    print("Q2 – cosine similarity:", round(sim, 1))


def q3_q4_ranking(query_vec):
    """Q3 & Q4 – rank documents by cosine similarity."""
    documents = [
        {"text": "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
         "section": "General course-related questions",
         "question": "Course - Can I still join the course after the start date?",
         "course": "data-engineering-zoomcamp"},
        {"text": "Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.",
         "section": "General course-related questions",
         "question": "Course - Can I follow the course after it finishes?",
         "course": "data-engineering-zoomcamp"},
        {"text": """The purpose of this document is to capture frequently asked technical questions
The exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  'Office Hours' live.
Subscribe to course public Google Calendar (it works from Desktop only).
Register before the course starts using this link.
Join the course Telegram channel with announcements.
Don't forget to register in DataTalks.Club's Slack and join the channel.""",
         "section": "General course-related questions",
         "question": "Course - When will the course start?",
         "course": "data-engineering-zoomcamp"},
        {"text": "You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.",
         "section": "General course-related questions",
         "question": "Course - What can I do before the course starts?",
         "course": "data-engineering-zoomcamp"},
        {"text": "Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.",
         "section": "General course-related questions",
         "question": "How can we contribute to the course?",
         "course": "data-engineering-zoomcamp"}
    ]
    model_name = "jinaai/jina-embeddings-v2-small-en"
    embedder = TextEmbedding(model_name=model_name)

    # Q3 – use only text field
    V_text = np.array(list(embedder.embed([d["text"] for d in documents])))
    sims_text = V_text.dot(query_vec)
    best_q3 = int(np.argmax(sims_text))
    print("Q3 – best doc index (text only):", best_q3)

    # Q4 – use question + text
    full_texts = [d["question"] + " " + d["text"] for d in documents]
    V_full = np.array(list(embedder.embed(full_texts)))
    sims_full = V_full.dot(query_vec)
    best_q4 = int(np.argmax(sims_full))
    print("Q4 – best doc index (question+text):", best_q4)


def q5_smallest_dimension():
    """Q5 – find smallest embedding dimension available in fastembed."""
    dims = [m["dim"] for m in TextEmbedding.list_supported_models() if "dim" in m]
    smallest = min(dims)
    print("Q5 – smallest model dimensionality:", smallest)
    return smallest


def q6_qdrant_demo(query_str):
    """Q6 – index ML Zoomcamp FAQ docs into in-memory Qdrant and query."""
    # pick the 384-dim BGE small
    model_name = "BAAI/bge-small-en"
    embedder = TextEmbedding(model_name=model_name)

    # download documents
    url = "https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json"
    docs = requests.get(url).json()
    records = []
    for course in docs:
        if course["course"] != "machine-learning-zoomcamp":
            continue
        for d in course["documents"]:
            records.append({"text": d["question"] + " " + d["text"], "payload": d})

    vectors = list(embedder.embed([r["text"] for r in records]))

    client = QdrantClient(":memory:")
    dim = len(vectors[0])
    client.recreate_collection(
        "ml_zoomcamp_faq",
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    client.upload_collection(
        collection_name="ml_zoomcamp_faq",
        vectors=vectors,
        payload=[r["payload"] for r in records],
    )

    q_vec = next(embedder.embed([query_str]))
    hit = client.search("ml_zoomcamp_faq", q_vec, limit=1)[0]
    print("Q6 – highest score:", round(hit.score, 2))


if __name__ == "__main__":
    query_embedding = q1_get_query_embedding()
    q2_similarity(query_embedding)
    q3_q4_ranking(query_embedding)
    q5_smallest_dimension()
    q6_qdrant_demo("I just discovered the course. Can I join now?") 