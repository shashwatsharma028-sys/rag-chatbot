import streamlit as st
import tempfile, json, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title='DocMind AI Pro+', page_icon='🧠', layout='wide')

st.markdown("""
<style>
.block-container{max-width:1450px;padding-top:1rem}
.hero{padding:2rem;border-radius:22px;background:linear-gradient(135deg,#0f172a,#4338ca);color:white;text-align:center;margin-bottom:1rem}
.card{padding:1rem;border:1px solid #e5e7eb;border-radius:18px;background:#ffffff08}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='hero'><h1>🧠 DocMind AI Pro+</h1><p>Financial Research Copilot • RAG • Analytics • Citations</p></div>", unsafe_allow_html=True)

for k,v in {'docs':[],'chunks':[],'vec':None,'mat':None,'chat':[]}.items():
    if k not in st.session_state: st.session_state[k]=v

@st.cache_resource
def llm(key): return ChatGroq(api_key=key, model='llama-3.1-8b-instant', temperature=0)

@st.cache_resource
def build(texts):
    vec=TfidfVectorizer(stop_words='english', max_features=5000)
    mat=vec.fit_transform(texts)
    return vec,mat

with st.sidebar:
    key=st.text_input('Groq API Key', type='password')
    mode=st.radio('Mode',['Chat','Compare','Dashboard','Summary'])
    files=st.file_uploader('Upload PDFs', type='pdf', accept_multiple_files=True)

if files:
    names=[f.name for f in files]
    if names!=st.session_state.docs:
        chunks=[]
        for f in files:
            with tempfile.NamedTemporaryFile(delete=False,suffix='.pdf') as t:
                t.write(f.read()); path=t.name
            docs=PyMuPDFLoader(path).load()
            parts=RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100).split_documents(docs)
            for p in parts: p.metadata['source']=f.name
            chunks.extend(parts)
        texts=[c.page_content for c in chunks]
        vec,mat=build(texts)
        st.session_state.docs=names; st.session_state.chunks=chunks; st.session_state.vec=vec; st.session_state.mat=mat
        st.success('Indexed documents successfully.')

c1,c2,c3=st.columns(3)
c1.metric('PDFs', len(st.session_state.docs))
c2.metric('Chunks', len(st.session_state.chunks))
c3.metric('Status', 'Ready' if key else 'Need API Key')

def search(q,k=4):
    if st.session_state.vec is None: return [],0
    sims=cosine_similarity(st.session_state.vec.transform([q]), st.session_state.mat)[0]
    ids=np.argsort(sims)[::-1][:k]
    docs=[st.session_state.chunks[i] for i in ids]
    return docs, round(float(np.mean([sims[i] for i in ids]))*100,1)

if mode=='Chat':
    q=st.chat_input('Ask about your documents...')
    for m in st.session_state.chat:
        with st.chat_message(m['r']): st.markdown(m['c'])
    if q and key:
        st.session_state.chat.append({'r':'user','c':q})
        docs,conf=search(q)
        ctx='\n\n'.join([d.page_content for d in docs])
        prompt=PromptTemplate.from_template('Use context to answer with bullet points. Include citations. Context:{c} Question:{q}')
        ans=(prompt|llm(key)|StrOutputParser()).invoke({'c':ctx,'q':q})
        cites='\n'.join([f"• {d.metadata.get('source','file')} p.{d.metadata.get('page',0)+1}" for d in docs])
        out=ans+f"\n\n**Sources**\n{cites}\n\n**Confidence:** {conf}%"
        st.session_state.chat.append({'r':'assistant','c':out})
        st.rerun()

elif mode=='Compare':
    if len(st.session_state.docs)>=2:
        a=st.selectbox('Doc A', st.session_state.docs)
        b=st.selectbox('Doc B', st.session_state.docs, index=1)
        topic=st.text_input('Compare Topic','revenue risk growth')
        if st.button('Run Comparison') and key:
            res=llm(key).invoke(f'Compare {a} and {b} on {topic}. Use table style bullets.')
            st.write(res.content if hasattr(res,'content') else str(res))
    else:
        st.info('Upload at least 2 PDFs.')

elif mode=='Dashboard':
    st.subheader('Auto KPI Dashboard')
    st.info('Upload annual reports and ask metrics like revenue, profit, debt, risk.')
    q=st.text_input('Metric Query','revenue trend and risks')
    if st.button('Generate') and key:
        docs,_=search(q,6)
        txt=' '.join([d.page_content[:1000] for d in docs])
        st.write((llm(key).invoke('Summarize KPIs from: '+txt).content))

else:
    st.subheader('Executive Summary')
    if st.button('Generate Summary') and key:
        docs,_=search('summary overview main findings',6)
        txt=' '.join([d.page_content[:1000] for d in docs])
        st.write((llm(key).invoke('Create executive summary with risks and opportunities: '+txt).content))

st.caption('Built by Shashwat Sharma • FAANG-level Portfolio Project')