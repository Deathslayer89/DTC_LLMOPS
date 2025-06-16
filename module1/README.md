# LLM ZoomCamp Module 1 Homework - ElasticSearch 

## Solutions

### Q1: ElasticSearch Build Hash
**Answer:** `dbcbbbd0bc4924cfeb28929dc05d82d662c527b7`
Retrieved by running `curl localhost:9200` and extracting the version.build_hash value from the cluster info.

### Q2: Indexing Function  
**Answer:** `index`
The `index` function is used to add documents to ElasticSearch.

### Q3: Search Score with Boosting
**Answer:** `44.50` (actual: 44.017967)
Searched "How do execute a command on a Kubernetes pod?" with question^4 boosting and best_fields type.

### Q4: Filtered Search Result
**Answer:** `How do I copy files from a different folder into docker container's working directory?`
Filtered by machine-learning-zoomcamp course and retrieved the 3rd result for Docker container query.

### Q5: Prompt Length
**Answer:** `1446`
Built a context from 3 search results and created a teaching assistant prompt template.

### Q6: Token Count
**Answer:** `320`
Used tiktoken with gpt-4o encoding to count tokens in the constructed prompt.

