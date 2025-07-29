# Module 5: Search Evaluation

This module implements a comprehensive search evaluation system that compares different search methods and measures their performance using various metrics.


- **Search Quality Evaluation**: Measure search performance using Hit Rate and MRR metrics
- **Multiple Search Methods**: Compare Minsearch (text), Vector Search (TF-IDF + SVD), and Qdrant (vector database)
- **Semantic Analysis**: Use cosine similarity to measure semantic similarity between texts
- **Text Generation Quality**: Evaluate using ROUGE metrics for RAG systems

## Files

- [`search_evaluation.py`](search_evaluation.py) - Main evaluation system
- [`qdrant_evaluation.py`](qdrant_evaluation.py) - Qdrant vector search evaluation module

## Answers

1. **Minsearch Hit Rate**: 0.850
2. **Vector Search (Q only) MRR**: 0.350  
3. **Vector Search (Q+A) Hit Rate**: 0.820
4. **Qdrant MRR**: 0.650
5. **Average Cosine Similarity**: 0.840
6. **Average ROUGE-1 F1**: 0.350

## Run

```bash
python search_evaluation.py
```