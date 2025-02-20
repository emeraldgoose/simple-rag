from io import StringIO
import pandas as pd
import streamlit as st

from backend.llm import get_answer, get_contexts
from backend.preprocess import load_data


st.title('Simple RAG')

def init_upload_file():
    st.session_state["document"] = None
    st.session_state["retriever_process"] = None

def show_retrieval(retrieval):
    tabs = st.tabs([f'{rank} ({score})' for rank,(_, score) in zip(range(1,len(retrieval)+1),retrieval)])
    for tab, (document, _) in zip(tabs, retrieval):
        with tab:
            st.markdown("### Page Content")
            st.markdown(document.page_content)
            if document.metadata['category'] == 'Table':
                origin_table = StringIO(document.metadata['origin_table'])
                table = pd.read_html(origin_table, header=0, index_col=0)[0]
                table = table.apply(lambda row: [str(el) for el in row])
                st.table(table)
            if document.metadata['category'] == 'CompositeElement' and 'text_as_html' in document.metadata.keys():
                origin_table = StringIO(document.metadata['text_as_html'])
                table = pd.read_html(origin_table, header=0, index_col=0)[0]
                table = table.apply(lambda row: [str(el) for el in row])
                st.table(table)


with st.sidebar:
    uploaded_file = st.sidebar.file_uploader("Choose a document", type=["pdf"],on_change=init_upload_file)
    if not uploaded_file:
        init_upload_file()
    
    completed = False
    if uploaded_file and st.session_state["document"] != uploaded_file.name:
        st.session_state["document"] = uploaded_file.name
        completed = load_data(uploaded_file.name)
    elif st.session_state["document"] == uploaded_file.name:
        completed = True

    with st.container(border=True):
        st.write('Configure Retriever')
        alpha = st.slider(
            label='Hybrid Search Alpha', 
            min_value=0.0, 
            max_value=1.0, 
            value=0.49, 
            step=0.01, 
            help='0 = lexical search | 1 = semantic search',
            disabled=(True if not completed else False)
        )
        retrieval_size = st.slider(label='Retrieval Size', min_value=1, max_value=5, value=3, step=1, disabled=(True if not completed else False))
        reranker = st.toggle("Reranker", disabled=(True if not completed else False))


with st.empty():
    if "history" not in st.session_state:
        st.session_state['history'] = []
    
    chat_room = st.container()
    if st.session_state['history']:
        for (name, chat_history) in st.session_state.history:
            chat_room.chat_message(name).write(chat_history)
    
    if prompt := st.chat_input("Say something"):
        chat_room.chat_message("user").write(prompt)
        
        index_name = st.session_state.document.split('.')[0]
        contexts, table_as_html, retrieval = get_contexts(
            query=prompt, 
            alpha=alpha, 
            k=retrieval_size, 
            index_name=index_name,
            reranker=reranker,
        )
        
        assistant = chat_room.chat_message("assistant")
        response = assistant.write_stream(
            get_answer(
                query=prompt, 
                contexts=contexts, 
                tables_html=table_as_html
            )
        )
        with assistant:
            with st.expander("References"):
                show_retrieval(retrieval)
        
        st.session_state['history'].append(('user', prompt))
        st.session_state['history'].append(('assistant',response))
