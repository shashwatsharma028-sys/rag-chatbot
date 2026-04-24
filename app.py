import os
import json
import tempfile
import numpy as np
import streamlit as st

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="DocMind AI",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------------------------------
# CACHE HELPERS
# ---------------------------------------------------
@st.cache_data(ttl=600)
def get_stock_data(ticker, period):
    import yfinance as yf
    stock = yf.Ticker(ticker)
    return stock.history(period=period), stock.info

@st.cache_resource
def get_llm(api_key):
    from langchain_groq import ChatGroq
    return ChatGroq(
        api_key=api_key,
        model="llama-3.1-8b-instant",
        temperature=0
    )

# ---------------------------------------------------
# STYLE
# ---------------------------------------------------
st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg,#667eea,#764ba2);
    padding:2rem;
    border-radius:14px;
    color:white;
    text-align:center;
    margin-bottom:2rem;
}
.metric-card{
    background:#f8f9ff;
    border:1px solid #e0e4ff;
    border-radius:12px;
    padding:1rem;
    text-align:center;
}
.metric-card h3{
    margin:0;
    color:#667eea;
}
.source-tag{
    background:#eef2ff;
    padding:4px 8px;
    border-radius:8px;
    margin:2px;
    display:inline-block;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HEADER
# ---------------------------------------------------
st.markdown("""
<div class="main-header">
<h1>🧠 DocMind AI</h1>
<p>AI-Powered Financial Document Intelligence System</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SIDEBAR
# ---------------------------------------------------
with st.sidebar:
    st.subheader("⚙️ Configuration")

    groq_key = st.text_input(
        "Groq API Key",
        type="password",
        value=os.getenv("GROQ_API_KEY", ""),
        placeholder="gsk_..."
    )

    uploaded_files = st.file_uploader(
        "Upload PDF Files",
        type=["pdf"],
        accept_multiple_files=True
    )

    mode = st.radio(
        "Mode",
        ["💬 Chat with PDFs", "📊 Compare Documents", "📈 Financial Analysis"]
    )

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------
defaults = {
    "chat_history": [],
    "chunks": [],
    "doc_chunks": {},
    "vectorizer": None,
    "tfidf_matrix": None,
    "indexed_files": [],
    "total_chunks": 0
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ---------------------------------------------------
# PDF INDEXING
# ---------------------------------------------------
def process_pdfs(files):
    from langchain_community.document_loaders import PyMuPDFLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from sklearn.feature_extraction.text import TfidfVectorizer

    all_chunks = []
    doc_chunks = {}

    progress = st.progress(0, text="Processing PDFs...")

    for i, file in enumerate(files):
        progress.progress(i / len(files), text=f"Reading {file.name}")

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(file.read())
            path = f.name

        loader = PyMuPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        chunks = splitter.split_documents(docs)

        for c in chunks:
            c.metadata["source"] = file.name

        doc_chunks[file.name] = chunks
        all_chunks.extend(chunks)

    texts = [c.page_content for c in all_chunks]

    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(texts)

    st.session_state.chunks = all_chunks
    st.session_state.doc_chunks = doc_chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf
    st.session_state.indexed_files = [f.name for f in files]
    st.session_state.total_chunks = len(all_chunks)

    progress.empty()
    st.success("✅ PDFs indexed successfully")

# ---------------------------------------------------
# RETRIEVAL
# ---------------------------------------------------
def retrieve_chunks(query, k=4):
    from sklearn.metrics.pairwise import cosine_similarity

    if st.session_state.vectorizer is None:
        return [], 0

    q = st.session_state.vectorizer.transform([query])
    scores = cosine_similarity(q, st.session_state.tfidf_matrix)[0]

    idx = np.argsort(scores)[::-1][:k]

    docs = [st.session_state.chunks[i] for i in idx]

    conf = round(float(np.mean([scores[i] for i in idx])) * 100, 1)

    return docs, conf

# ---------------------------------------------------
# AUTO INDEX
# ---------------------------------------------------
new_files = [f.name for f in uploaded_files] if uploaded_files else []

if uploaded_files and new_files != st.session_state.indexed_files:
    process_pdfs(uploaded_files)

# ---------------------------------------------------
# SIDEBAR DOC STATUS
# ---------------------------------------------------
if st.session_state.indexed_files:
    with st.sidebar:
        st.markdown("---")
        st.write("Indexed Files")
        for x in st.session_state.indexed_files:
            st.write("✅", x)

# ---------------------------------------------------
# MODE 1 CHAT
# ---------------------------------------------------
if mode == "💬 Chat with PDFs":

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask about your PDFs...")

    if question:

        if not groq_key:
            st.warning("Please add GROQ key.")
            st.stop()

        if not st.session_state.chunks:
            st.warning("Upload PDF first.")
            st.stop()

        st.session_state.chat_history.append(
            {"role": "user", "content": question}
        )

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                docs, conf = retrieve_chunks(question)

                pdf_context = "\n\n".join(
                    [d.page_content for d in docs]
                )

                try:
                    from langchain_community.tools import DuckDuckGoSearchRun
                    web_context = DuckDuckGoSearchRun().run(question)
                except:
                    web_context = "No web results."

                from langchain_core.prompts import PromptTemplate
                from langchain_core.output_parsers import StrOutputParser

                prompt = PromptTemplate.from_template("""
Use PDF context + web context.

PDF:
{pdf}

WEB:
{web}

Question:
{q}

Answer clearly.
""")

                llm = get_llm(groq_key)

                chain = prompt | llm | StrOutputParser()

                answer = chain.invoke({
                    "pdf": pdf_context,
                    "web": web_context,
                    "q": question
                })

                st.markdown(answer)

                st.markdown(f"**Confidence:** {conf}%")

                st.session_state.chat_history.append(
                    {"role": "assistant", "content": answer}
                )

# ---------------------------------------------------
# MODE 2 COMPARE
# ---------------------------------------------------
elif mode == "📊 Compare Documents":

    if len(st.session_state.indexed_files) < 2:
        st.info("Upload at least 2 PDFs.")
    else:
        col1, col2 = st.columns(2)

        with col1:
            d1 = st.selectbox(
                "Document 1",
                st.session_state.indexed_files
            )

        with col2:
            d2 = st.selectbox(
                "Document 2",
                st.session_state.indexed_files
            )

        topic = st.text_input("Topic to compare")

        if st.button("Compare") and topic:

            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
            from langchain_core.prompts import PromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            def top_context(docname):
                chunks = st.session_state.doc_chunks[docname]
                texts = [c.page_content for c in chunks]

                vec = TfidfVectorizer()
                mat = vec.fit_transform(texts)

                q = vec.transform([topic])

                scores = cosine_similarity(q, mat)[0]
                idx = np.argsort(scores)[::-1][:4]

                return "\n\n".join(
                    [chunks[i].page_content for i in idx]
                )

            c1 = top_context(d1)
            c2 = top_context(d2)

            llm = get_llm(groq_key)

            prompt = PromptTemplate.from_template("""
Compare documents on topic: {topic}

Doc1:
{c1}

Doc2:
{c2}

Give similarities, differences, winner.
""")

            ans = (prompt | llm | StrOutputParser()).invoke({
                "topic": topic,
                "c1": c1,
                "c2": c2
            })

            st.markdown(ans)

# ---------------------------------------------------
# MODE 3 FINANCE
# ---------------------------------------------------
else:

    import plotly.graph_objects as go
    import plotly.express as px

    ticker = st.text_input("Ticker", value="AAPL")
    period = st.selectbox(
        "Period",
        ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        index=3
    )

    if st.button("Load Chart"):

        with st.spinner("Loading chart..."):

            hist, info = get_stock_data(ticker, period)

            if hist.empty:
                st.error("Invalid ticker")
            else:
                st.metric(
                    "Current Price",
                    round(hist["Close"].iloc[-1], 2)
                )

                fig = go.Figure()

                fig.add_trace(go.Candlestick(
                    x=hist.index,
                    open=hist["Open"],
                    high=hist["High"],
                    low=hist["Low"],
                    close=hist["Close"]
                ))

                fig.update_layout(height=500)

                st.plotly_chart(fig, use_container_width=True)

                vol = px.bar(
                    hist,
                    x=hist.index,
                    y="Volume"
                )

                st.plotly_chart(vol, use_container_width=True)