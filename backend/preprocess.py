from itertools import chain
from tempfile import TemporaryDirectory
import sqlite3

from langchain.schema import Document
from langchain_community.vectorstores import OpenSearchVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from llama_cloud_services import LlamaParse
from llama_index.core import schema as llama_index_schema

from backend import get_secret_key
from backend.llm import get_summary
from backend.retriever import get_opensearch_client

def get_connection():
    conn = sqlite3.connect('db/database.db')
    return conn

def to_markdown(obj, path):
    with open(file=path, mode="w") as f:
        f.write(obj)

def llamaparse_to_db(conn, table, llamaparse):
    insert_data = []
    for document in llamaparse:
        id_ = document.id_
        text_resource = document.text_resource.text
        insert_data.append((id_, text_resource))

    cur = conn.cursor()
    cur.executemany(f'insert into {table} values (?, ?)', insert_data)
    conn.commit()

def create_table(conn, table):
    cur = conn.cursor()
    tables = cur.execute('select name from sqlite_master where type="table"').fetchall()[0]
    if table in tables:
        return True
    
    cur.execute(f'create table if not exists {table} (id text, resoource text)')
    return False

def get_documents_from_db(conn, table):
    cur = conn.cursor()
    tables = cur.execute(f'select * from {table}').fetchall()
    llamaparse = [llama_index_schema.Document(doc_id=id_,text=text) for id_,text in tables]
    return llamaparse

def convert_markdown(filepath):
    llamaparse = LlamaParse(
        result_type="markdown",
        api_key=get_secret_key('LLAMA_CLOUD_API_KEY')
    ).load_data(f'documents/{filepath}')
    return llamaparse

def parse(llamaparse):
    texts, tables = [], []
    with TemporaryDirectory() as temp_dir:
        for i, res_ in enumerate(llamaparse):
            markdown_path = f"{temp_dir}/parsed_llamaparse_{i}.md"
            to_markdown(res_.get_content('text'), markdown_path)

            loader = UnstructuredMarkdownLoader(
                file_path=markdown_path,
                mode="elements",
                chunking_strategy="by_title",
                strategy="hi_res",
                max_characters=4096,
                new_after_n_chars=4000,
                combine_text_under_n_chars=2000,
            )

            docs_llamaparse = loader.load()
            if 'text_as_html' in docs_llamaparse[0].metadata.keys():
                table_doc = Document(page_content=docs_llamaparse[0].metadata['text_as_html'])
                table_doc.metadata['category'] = 'Table'
                tables.append(table_doc)
            else:
                texts.append(docs_llamaparse[0])
    
    return texts, tables

def get_create_index_body():
    index_body = {
        'settings': {
            'analysis': {
                'analyzer': {
                    'my_analyzer': {
                        'char_filter':['html_strip'],
                        'tokenizer': 'nori',
                        'filter': ['nori_filter'],
                        'type': 'custom'
                    }
                },
                'tokenizer': {
                    'nori': {
                        'decompound_mode': 'mixed',
                        'discard_punctuation': 'true',
                        'type': 'nori_tokenizer'
                    }
                },
                "filter": {
                    "nori_filter": {
                        "type": "nori_part_of_speech",
                        "stoptags": [
                            "J", "XSV", "E", "IC","MAJ","NNB",
                            "SP", "SSC", "SSO",
                            "SC","SE","XSN","XSV",
                            "UNA","NA","VCP","VSV",
                            "VX"
                        ]
                    }
                }
            },
            'index': {
                'knn': True,
                'knn.space_type': 'cosinesimil'
            }
        },
        'mappings': {
            'properties': {
                'metadata': {
                    'properties': {
                        'category': {'type':'text'},
                        'origin_table': {'type':'text'},
                    }
                },
                'text': {
                    'analyzer': 'my_analyzer',
                    'search_analyzer': 'my_analyzer',
                    'type': 'text'
                },
                'vector_field': {
                    'type': 'knn_vector',
                    'dimension': "768"
                }
            }
        }
    }
    return index_body

def create_index(client, index_name):
    index_exists = client.opensearch.indices.exists(index_name)
    if index_exists:
        response = client.opensearch.indices.delete(index_name)
        if not response['acknowledged']:
            raise Exception('Failed delete_index')
    
    index_body = get_create_index_body()
    response = client.opensearch.indices.create(index_name, body=index_body)
    return response['acknowledged']
    
def store_vector_store(index_name, docs):
    opensearch_url = get_secret_key('OPENSEARCH_URL')
    opensearch_id = get_secret_key('OPENSEARCH_ID')
    opensearch_password = get_secret_key('OPENSEARCH_PASSWORD')
    google_api_key = get_secret_key('GOOGLE_API_KEY')

    llm_emb = GoogleGenerativeAIEmbeddings(
        model='models/text-embedding-004',
        google_api_key=google_api_key
    )
    vector_db = OpenSearchVectorSearch(
        index_name=index_name,
        opensearch_url=opensearch_url,
        embedding_function=llm_emb,
        http_auth=(opensearch_id, opensearch_password),
        is_aoss=False,
        engine="faiss",
        space_type="l2",
        bulk_size=100000,
        timeout=60,
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False
    )

    vector_db.add_documents(
        documents=docs,
        vector_field='vector_field',
        bulk_size=100000,
    )

def load_data(filepath):
    filename = filepath.split('.')[0]
    conn = get_connection()
    table_exists = create_table(conn, filename)
    if table_exists:
        llamaparse = get_documents_from_db(conn, filename)
    else:
        llamaparse = convert_markdown(filepath)
        llamaparse_to_db(conn, filename, llamaparse)

    texts, tables = parse(llamaparse)
    table_summaries = get_summary(tables, api_key=get_secret_key('GOOGLE_API_KEY'))

    table_preprocessed = []
    for origin, summary in zip(tables, table_summaries):
        metadata = origin.metadata
        metadata['origin_table'] = origin.page_content
        doc = Document(
            page_content=summary,
            metadata=metadata
        )
        table_preprocessed.append(doc)

    docs = list(chain(texts,table_preprocessed))

    client = get_opensearch_client()
    create_index(client, filename)
    store_vector_store(filename, docs)

    return True