from langchain.schema import StrOutputParser
from langchain_core.prompts import HumanMessagePromptTemplate, SystemMessagePromptTemplate, ChatPromptTemplate
from langchain_core.tracers import ConsoleCallbackHandler
from langchain_google_genai import GoogleGenerativeAI

from backend.retriever import get_opensearch_client

def get_llm(api_key: str):
    llm = GoogleGenerativeAI(model='gemini-2.0-flash-001', api_key=api_key, temperature=0)
    return llm

def get_system_template():
    instruction = """You are a master answer bot designed to answer user's question.\nI'm going to give you contexts which consist of texts, tables and images.\nRead the contexts carefully, because I'm going to ask you a question about it."""
    system_template = SystemMessagePromptTemplate.from_template(instruction)
    return system_template

def get_human_template(is_table: bool):
    template = """Here is the contexts as texts: <context>{contexts}</context>\nTABLE_PROMPT\n\nFirst, find a few paragraphs or sentences from the contexts that are most relevant to answering the question.\nThen, answer the question as much as you can.\n\nSkip the preamble and go straight into the answer.\nDon't insert any XML tag such as <contexts> and </contexts> when answering.\nAnswer in Korean.\n\nHere is the question: <question>{question}</question>\n\nIf the question cannot be answered by the contexts. say "No relevant contexts"."""
    table_prompt = """Here is the contexts as tables (table as html): <table_summary>{tables_html}</tables_summary>"""
    if is_table:
        template = template.replace("TABLE_PROMPT",table_prompt)
    else:
        template = template.replace("TABLE_PROMPT",'')
    
    human_template = HumanMessagePromptTemplate.from_template(template)
    return human_template

def get_summary(tables, api_key):
    llm = get_llm(api_key)
    system_prompt = "You are an assistant tasked with describing table and image."
    human_prompt = [
        {
            "type": "text",
            "text": '''Here is the table: {table}\nGiven table, give a concise summary.\nDon't insert any XML tag such as <table> and </table> when answering.\nWrite in Korean.'''
        },
    ]
    system_message_template = SystemMessagePromptTemplate.from_template(system_prompt)
    human_message_template = HumanMessagePromptTemplate.from_template(human_prompt)
    prompt = ChatPromptTemplate.from_messages(
        [
            system_message_template,
            human_message_template
        ]
    )

    summarize_chain = {"table": lambda x:x} | prompt | llm | StrOutputParser()
    table_summaries = summarize_chain.batch(tables)
    return table_summaries

def new_retrieval(reranked, retrieval):
    arr = [(doc,r['output'][0]['data'][0]) for r, (doc, _) in zip(reranked,retrieval)]
    arr.sort(key=lambda x: x[1], reverse=True)
    return arr

def get_contexts(**kwargs):
    query = kwargs.get('query')
    alpha = kwargs.get('alpha', 0.49)
    index_name = kwargs.get('index_name')
    k = kwargs.get('k')
    reranker = kwargs.get('reranker',False)

    retriever = get_opensearch_client()
    retrieval = retriever.get_retrieval(
        query=query, 
        alpha=alpha, 
        index_name=index_name, 
        k=int(k*1.5) if reranker else k
    )
    
    if reranker:
        reranked = retriever.rerank_documents(query, retrieval)
        retrieval = new_retrieval(reranked, retrieval)[:k]

    contexts = '\n\n'.join([doc.page_content for (doc, _) in retrieval])

    tables = []
    for doc,_ in retrieval:
        if doc.metadata['category'] in ('CompositeElement','Table'):
            if 'text_as_html' in doc.metadata.keys():
                tables.append(doc.metadata['text_as_html'])
            elif 'origin_table' in doc.metadata.keys():
                tables.append(doc.metadata['origin_table'])
    table_as_html = '\n\n'.join(tables) if tables else ''
    
    return contexts, table_as_html, retrieval
    

def get_answer(**kwargs):
    llm = get_llm(kwargs.get('api_key'))
    contexts = kwargs.get('contexts','')
    tables_html = kwargs.get('tables_html', '')

    system_template = get_system_template()
    human_template = get_human_template(is_table=True)

    prompt = ChatPromptTemplate.from_messages([system_template, human_template])

    invoke_args = {
        'question': kwargs.get('query'),
        'contexts': contexts,
        'tables_html': tables_html
    }
    verbose = kwargs.get('verbose', False)

    chain = prompt | llm | StrOutputParser()
    response = chain.stream(
        invoke_args,
        config={'callbacks': [ConsoleCallbackHandler()]} if verbose else {}
    )

    for chunk in response:
        yield chunk
