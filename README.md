# Simple-RAG

## Requirements
```
pip install -r requirements.txt
```
## backend/secret.env
```
LLAMA_CLOUD_API_KEY=''
GOOGLE_API_KEY=''
OPENSEARCH_HOST=localhost
OPENSEARCH_PORT=9200
OPENSEARCH_URL=https://localhost:9200
OPENSEARCH_ID=''
OPENSEARCH_PASSWORD=''
OPENSEARCH_RERANK_MODEL_ID=''
```
### Run
```
cd infra
docker compose up -d
```
```
streamlit run app.py --server.port 8080
```