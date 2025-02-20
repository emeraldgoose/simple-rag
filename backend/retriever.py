from typing_extensions import List, Tuple

from opensearchpy import OpenSearch
from langchain.schema import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from backend import get_secret_key

class OpenSearchRetriever:
    def __init__(self, hosts, **kwargs):
        self.opensearch = OpenSearch(
            hosts,
            http_compress=kwargs.get('http_compress',True),
            http_auth=kwargs.get('http_auth'),
            use_ssl=kwargs.get('use_ssl',True),
            verify_certs=kwargs.get('verify_certs',False)
        )
        self.llm_emb = GoogleGenerativeAIEmbeddings(
            model='models/text-embedding-004',
            google_api_key=get_secret_key('GOOGLE_API_KEY')
        )

    def get_document(self, query, index_name):
        response = self.opensearch.search(body=query, index=index_name)
        return response

    def get_query(self, **kwargs):
        search_type = kwargs.get('search_type', 'lexical')

        # lexcial query
        if search_type == 'lexical':
            template = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "match": {
                                    "text": {
                                        "query": f"{kwargs.get("query")}",
                                        "minimum_should_match": f"{kwargs.get("minimum_should_match",0)}%",
                                        "operator": "or"
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                }
            }

        # semantic query
        if search_type == 'semantic':
            template = {
                "query": {
                    "bool": {
                        "must": [
                            {
                                "knn": {
                                    f"{kwargs.get("vector_field")}": {
                                        "vector": kwargs.get("vector"),
                                        "k": kwargs.get("k")
                                    }
                                }
                            }
                        ],
                        "filter": []
                    }
                }
            }

        return template

    def get_lexical_search(self, **kwargs):
        query = self.get_query(
            query=kwargs.get('query'), 
            minimum_should_match=kwargs.get('minimum_should_match',0)
        )
        query['size'] = kwargs.get('k',3)
        response = self.get_document(
            query, 
            index_name=kwargs.get('index_name')
        )
        return response
    
    def get_semantic_search(self, **kwargs):
        query = self.get_query(
            query=kwargs.get('query'),
            search_type='semantic',
            vector_field=kwargs.get('vector_field'),
            vector=kwargs.get('vector'),
            k=kwargs.get('k',3)
        )
        query['size'] = kwargs.get('k',3)
        response = self.get_document(
            query=query,
            index_name=kwargs.get('index_name')
        )
        return response

    def normalize_score(self, response):
        hits = response['hits']['hits']
        max_score = response['hits']['max_score']
        for hit in hits:
            hit['_score'] = float(hit['_score']) / max_score
        response['hits']['max_score'] = hits[0]['_score']
        response['hits']['hits'] = hits
        return response
    
    def preprocess(self, response):
        results = []
        for res in response['hits']['hits']:
            metadata = res['_source']['metadata']
            metadata['id'] = res['_id']
            
            doc = Document(page_content=res['_source']['text'], metadata=metadata)
            results.append((doc, res['_source']))
        return results
    
    def get_ensemble_results(self, doc_lists, weights, c=60, k=3):
        documents = set()
        content_to_document = {}
        for doc_list in doc_lists:
            for (doc, _) in doc_list:
                content = doc.page_content
                documents.add(content)
                content_to_document[content] = doc
        
        hybrid_score_dic = {content: 0.0 for content in documents}

        for doc_list, weight in zip(doc_lists, weights):
            for rank, (doc, score) in enumerate(doc_list, start=1):
                content = doc.page_content
                score = weight * (1 / (rank + c))
                hybrid_score_dic[content] += score

        sorted_documents = sorted(hybrid_score_dic.items(), key=lambda x: -x[1])

        sorted_docs = [
            (content_to_document[doc_id], hybrid_score) for (doc_id, hybrid_score) in sorted_documents
        ]

        return sorted_docs[:k]

    def get_retrieval(self, query: str, alpha: float, index_name: str, k: int=3):
        run_lexical = alpha < 1.0
        run_semantic = alpha > 0.0
        doc_lists = []
        weights = [1-alpha, alpha]
        
        if run_lexical:
            lexical_response = self.get_lexical_search(
                query=query,
                minimum_should_match=0,
                k=k,
                index_name=index_name
            )
            response = self.normalize_score(lexical_response)
            lexical_results = self.preprocess(response)
        else:
            lexical_results = []
        doc_lists.append(lexical_results)

        if run_semantic:
            vector = self.llm_emb.embed_query(query)
            semantic_response = self.get_semantic_search(
                query=query,
                vector_field='vector_field',
                vector=vector,
                k=k,
                index_name=index_name
            )
            response = self.normalize_score(semantic_response)
            semantic_results = self.preprocess(response)
        else:
            semantic_results = []
        doc_lists.append(semantic_results)
        
        retrieval = self.get_ensemble_results(doc_lists, weights, k=k)
        return retrieval

    def rerank_documents(self, query: str, retrieval: List[Tuple[Document, float]]):
        model_id = get_secret_key('OPENSEARCH_RERANK_MODEL_ID')
        text_docs = [doc.page_content for doc, _ in retrieval]

        payload = {
            'query_text': query,
            'text_docs': text_docs
        }
        response = self.opensearch.transport.perform_request(
            method='POST',
            url=f'/_plugins/_ml/models/{model_id}/_predict',
            body=payload,
        )
        
        response = response['inference_results']
        return response


def get_opensearch_client():
    host = get_secret_key('OPENSEARCH_HOST')
    port = get_secret_key('OPENSEARCH_PORT')
    id_ = get_secret_key('OPENSEARCH_ID')
    password = get_secret_key('OPENSEARCH_PASSWORD')
    
    client = OpenSearchRetriever(
        hosts=[{'host': host, 'port': port}],
        http_compress=True,
        http_auth=(id_, password),
        use_ssl=True,
        verify_certs=False
    )
    
    return client
