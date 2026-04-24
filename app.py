import streamlit as st
import tempfile
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title='DocMind AI Pro', page_icon='🧠', layout='wide')

st.markdown('''
<style>
.main{padding-top:1rem}
.block-container{padding-top:1rem;max-width:1400px}
.hero{background:linear-gradient(135deg,#667eea,#764ba2);padding:2rem;border-radius:20px;color:white;text-align:center;margin-bottom:1rem}
.card{background:#11182722;border:1px solid #e5e7eb33;padding:1rem;border-radius:18px}
.metric{padding:1rem;border-radius:16px;background:#f8fafc;border:1px solid #e5e7eb}
</style>
''', unsafe_allow_html=True)

st.markdown("""<div class='hero'><h1>🧠 DocMind AI Pro</h1><p>RAG-Powered Financial Document Intelligence Platform</p></div>""", unsafe_allow_html=True)

for k,v in {
'chat_history':[], 'chunks':[], 'indexed_files':[], 'vectorizer':None, 'matrix':None
}.items():
    if k not in st.session_state:
        st.session_state[k]=v

@st.cache_resource
def load_llm(key):
    return ChatGroq(api_key=key, model='llama-3.1-8b-instant', temperature=0)

@st.cache_resource
def build_index(texts):
    vec=TfidfVectorizer(stop_words='english')
    mat=vec.fit_transform(texts)
    return vec,mat

with st.sidebar:
    st.header('⚙️ Settings')
    api_key=st.text_input('Groq API Key', type='password')
    mode=st.radio('Mode',['💬 Chat','📊 Compare','📈 Finance'])
    files=st.file_uploader('Upload PDFs', type=['pdf'], accept_multiple_files=True)

if files and api_key:
    names=[f.name for f in files]
    if names!=st.session_state.indexed_files:
        all_chunks=[]
        with st.spinner('Indexing documents...'):
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as tmp:
                    tmp.write(f.read())
                    path=tmp.name
                docs=PyMuPDFLoader(path).load()
                splitter=RecursiveCharacterTextSplitter(chunk_size=600,chunk_overlap=80)
                chunks=splitter.split_documents(docs)
                for c in chunks:
                    c.metadata['source']=f.name
                all_chunks.extend(chunks)
        texts=[c.page_content for c in all_chunks]
        vec,mat=build_index(texts)
        st.session_state.chunks=all_chunks
        st.session_state.indexed_files=names
        st.session_state.vectorizer=vec
        st.session_state.matrix=mat
        st.success(f'Indexed {len(names)} file(s) successfully.')

col1,col2,col3=st.columns(3)
with col1: st.metric('PDFs', len(st.session_state.indexed_files))
with col2: st.metric('Chunks', len(st.session_state.chunks))
with col3: st.metric('Status', 'Ready' if api_key else 'API Key Needed')

def retrieve(q,k=4):
    if st.session_state.vectorizer is None:
        return [],0
    qv=st.session_state.vectorizer.transform([q])
    sims=cosine_similarity(qv, st.session_state.matrix)[0]
    idx=np.argsort(sims)[::-1][:k]
    docs=[st.session_state.chunks[i] for i in idx]
    conf=round(float(np.mean([sims[i] for i in idx]))*100,1)
    return docs,conf

if mode=='💬 Chat':
    st.subheader('Chat with your PDFs')
    for msg in st.session_state.chat_history:
        with st.chat_message(msg['role']):
            st.markdown(msg['content'])
    q=st.chat_input('Ask anything about your PDFs...')
    if q and api_key:
        st.session_state.chat_history.append({'role':'user','content':q})
        with st.chat_message('user'): st.markdown(q)
        with st.chat_message('assistant'):
            docs,conf=retrieve(q)
            ctx='\n\n'.join([d.page_content for d in docs])
            llm=load_llm(api_key)
            prompt=PromptTemplate.from_template('Answer from context. Mention sources if possible. Context:{context} Question:{question} Answer:')
            ans=(prompt|llm|StrOutputParser()).invoke({'context':ctx,'question':q})
            st.markdown(ans)
            st.progress(min(int(conf),100))
            st.caption(f'Confidence: {conf}%')
            st.session_state.chat_history.append({'role':'assistant','content':ans})

elif mode=='📊 Compare':
    st.subheader('Compare Two Documents')
    if len(st.session_state.indexed_files)>=2:
        a=st.selectbox('Document 1', st.session_state.indexed_files)
        b=st.selectbox('Document 2', st.session_state.indexed_files, index=1)
        topic=st.text_input('Topic', 'revenue growth risk')
        if st.button('Compare') and api_key:
            llm=load_llm(api_key)
            res=llm.invoke(f'Compare {a} and {b} on {topic} in concise bullet points.')
            st.write(res.content if hasattr(res,'content') else str(res))
    else:
        st.info('Upload at least 2 PDFs.')

else:
    st.subheader('Financial Dashboard')
    ticker=st.text_input('Ticker', 'AAPL')
    if st.button('Load Finance'):
        import yfinance as yf
        import plotly.graph_objects as go
        data=yf.Ticker(ticker).history(period='6mo')
        fig=go.Figure()
        fig.add_scatter(x=data.index,y=data['Close'],mode='lines',name='Close')
        fig.update_layout(height=450)
        st.plotly_chart(fig,use_container_width=True)

st.markdown('---')
st.caption('Built by Shashwat Sharma • Deployed Portfolio Project')