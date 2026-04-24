import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
import tempfile
import numpy as np
import pymupdf4llm
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import re

st.set_page_config(
    page_title="DocMind AI — Financial Analyst",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p { opacity: 0.9; margin: 0.5rem 0 0; font-size: 1.1rem; }
    .feature-badge {
        display: inline-block;
        background: rgba(255,255,255,0.2);
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        margin: 4px;
        color: white;
    }
    .metric-card {
        background: #f8f9ff;
        border: 1px solid #e0e4ff;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .metric-card h3 { color: #667eea; font-size: 1.8rem; margin: 0; }
    .metric-card p { color: #666; font-size: 0.85rem; margin: 0.2rem 0 0; }
    .source-tag {
        background: #f0f4ff;
        border: 1px solid #c7d2fe;
        border-radius: 6px;
        padding: 2px 8px;
        font-size: 0.8rem;
        color: #4338ca;
        margin: 2px;
        display: inline-block;
    }
    .compare-header {
        background: #f8f9ff;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin-bottom: 1rem;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .sidebar-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🧠 DocMind AI</h1>
    <p>AI-Powered Financial Document Intelligence System</p>
    <div style="margin-top: 1rem;">
        <span class="feature-badge">📄 Multi-PDF</span>
        <span class="feature-badge">🌐 Web Search</span>
        <span class="feature-badge">🔍 OCR Support</span>
        <span class="feature-badge">📊 Compare Docs</span>
        <span class="feature-badge">📈 Live Stock Charts</span>
        <span class="feature-badge">💰 Financial Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Configuration</p>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...")
    st.markdown("---")
    st.markdown('<p class="sidebar-title">📂 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="sidebar-title">🎮 Mode</p>', unsafe_allow_html=True)
    mode = st.radio("", ["💬 Chat with PDFs", "📊 Compare Documents", "📈 Financial Analysis"], label_visibility="collapsed")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "doc_chunks" not in st.session_state:
    st.session_state.doc_chunks = {}
if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0

new_files = [f.name for f in uploaded_files] if uploaded_files else []
if new_files != st.session_state.indexed_files and uploaded_files and groq_key:
    progress = st.progress(0, text="Starting indexing...")
    all_chunks = []
    doc_chunks = {}

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress((i) / len(uploaded_files), text=f"Processing {uploaded_file.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        total_text = " ".join([doc.page_content for doc in documents]).strip()
        if len(total_text) < 100:
            md_text = pymupdf4llm.to_markdown(tmp_path)
            documents = [Document(page_content=md_text, metadata={"source": uploaded_file.name, "page": 0})]
            st.info(f"🔍 Scanned PDF detected — OCR applied to {uploaded_file.name}")

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
        doc_chunks[uploaded_file.name] = chunks
        all_chunks.extend(chunks)

    progress.progress(0.8, text="Building vector database...")
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    vectorstore = Chroma.from_documents(all_chunks, embeddings)

    st.session_state.vectorstore = vectorstore
    st.session_state.embeddings = embeddings
    st.session_state.indexed_files = new_files
    st.session_state.doc_chunks = doc_chunks
    st.session_state.chat_history = []
    st.session_state.total_chunks = len(all_chunks)

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.success(f"✅ Successfully indexed {len(uploaded_files)} PDF(s)!")

if st.session_state.indexed_files:
    with st.sidebar:
        st.markdown("---")
        st.markdown('<p class="sidebar-title">📋 Indexed Documents</p>', unsafe_allow_html=True)
        for name in st.session_state.indexed_files:
            st.markdown(f"✅ {name}")
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{len(st.session_state.indexed_files)}</h3><p>PDFs</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{st.session_state.total_chunks}</h3><p>Chunks</p></div>', unsafe_allow_html=True)

if mode == "📈 Financial Analysis":
    st.markdown('<div class="compare-header"><h3>📈 Financial Analysis Dashboard</h3><p>Live stock charts + financial metrics extracted from your documents</p></div>', unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["📈 Live Stock Chart", "💰 Financial Metrics from PDF"])

    with tab1:
        st.subheader("Live Stock Price Chart")
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter stock ticker", placeholder="e.g. AAPL, TSLA, GOOGL", value="AAPL")
        with col2:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

        if st.button("📊 Load Chart"):
            with st.spinner(f"Fetching data for {ticker}..."):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    info = stock.info

                    if hist.empty:
                        st.error("Invalid ticker symbol. Please try again.")
                    else:
                        m1, m2, m3, m4 = st.columns(4)
                        current_price = round(hist['Close'].iloc[-1], 2)
                        prev_price = round(hist['Close'].iloc[-2], 2)
                        change = round(current_price - prev_price, 2)
                        change_pct = round((change / prev_price) * 100, 2)
                        high_52w = round(hist['Close'].max(), 2)
                        low_52w = round(hist['Close'].min(), 2)

                        with m1:
                            st.metric("Current Price", f"${current_price}", f"{change_pct}%")
                        with m2:
                            st.metric("52W High", f"${high_52w}")
                        with m3:
                            st.metric("52W Low", f"${low_52w}")
                        with m4:
                            market_cap = info.get('marketCap', 0)
                            st.metric("Market Cap", f"${round(market_cap/1e9, 1)}B" if market_cap else "N/A")

                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(
                            x=hist.index,
                            open=hist['Open'],
                            high=hist['High'],
                            low=hist['Low'],
                            close=hist['Close'],
                            name=ticker,
                            increasing_line_color='#00C851',
                            decreasing_line_color='#ff4444'
                        ))
                        fig.add_trace(go.Scatter(
                            x=hist.index,
                            y=hist['Close'].rolling(20).mean(),
                            name='20-day MA',
                            line=dict(color='#667eea', width=1.5)
                        ))
                        fig.update_layout(
                            title=f"{ticker} Stock Price — {period}",
                            xaxis_title="Date",
                            yaxis_title="Price (USD)",
                            template="plotly_white",
                            height=500,
                            xaxis_rangeslider_visible=False
                        )
                        st.plotly_chart(fig, use_container_width=True)

                        vol_fig = px.bar(
                            hist, x=hist.index, y='Volume',
                            title=f"{ticker} Trading Volume",
                            color='Volume',
                            color_continuous_scale='Blues'
                        )
                        vol_fig.update_layout(template="plotly_white", height=250)
                        st.plotly_chart(vol_fig, use_container_width=True)

                except Exception as e:
                    st.error(f"Error fetching data: {e}")

    with tab2:
        st.subheader("Financial Metrics from Your PDF")
        if not st.session_state.vectorstore:
            st.info("👆 Please upload a financial PDF first (annual report, 10-K, etc.)")
        else:
            if st.button("🔍 Extract Financial Data"):
                with st.spinner("Extracting financial metrics from document..."):
                    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 6})

                    finance_prompt = PromptTemplate.from_template("""Extract financial metrics from the context below.
Return ONLY a JSON object with these exact keys (use null if not found):
{{
  "company_name": "...",
  "revenues": [{{"year": "...", "value": ...}}],
  "net_income": [{{"year": "...", "value": ...}}],
  "gross_profit": [{{"year": "...", "value": ...}}],
  "key_insights": ["...", "...", "..."]
}}
All monetary values should be in millions (numbers only, no symbols).

Context: {context}""")

                    docs = retriever.invoke("revenue net income gross profit financial results annual")
                    context = "\n\n".join([doc.page_content for doc in docs])

                    chain = finance_prompt | llm | StrOutputParser()
                    result = chain.invoke({"context": context})

                    try:
                        import json
                        clean = result.strip()
                        if "```" in clean:
                            clean = clean.split("```")[1]
                            if clean.startswith("json"):
                                clean = clean[4:]
                        data = json.loads(clean)

                        if data.get("company_name"):
                            st.markdown(f"### {data['company_name']} — Financial Overview")

                        if data.get("revenues") and len(data["revenues"]) > 0:
                            rev_data = [r for r in data["revenues"] if r.get("value")]
                            if rev_data:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(
                                    x=[r["year"] for r in rev_data],
                                    y=[r["value"] for r in rev_data],
                                    name="Revenue",
                                    marker_color="#667eea"
                                ))
                                if data.get("net_income"):
                                    ni_data = [r for r in data["net_income"] if r.get("value")]
                                    if ni_data:
                                        fig.add_trace(go.Bar(
                                            x=[r["year"] for r in ni_data],
                                            y=[r["value"] for r in ni_data],
                                            name="Net Income",
                                            marker_color="#764ba2"
                                        ))
                                fig.update_layout(
                                    title="Revenue vs Net Income (in Millions)",
                                    xaxis_title="Year",
                                    yaxis_title="Amount (USD Millions)",
                                    template="plotly_white",
                                    barmode="group",
                                    height=400
                                )
                                st.plotly_chart(fig, use_container_width=True)

                        if data.get("key_insights"):
                            st.markdown("### 💡 Key Financial Insights")
                            for insight in data["key_insights"]:
                                if insight:
                                    st.markdown(f"- {insight}")

                    except Exception:
                        st.markdown("### 📊 Extracted Financial Information")
                        st.markdown(result)

elif mode == "📊 Compare Documents":
    st.markdown('<div class="compare-header"><h3>📊 Document Comparison</h3><p>Compare two documents on any topic using semantic similarity</p></div>', unsafe_allow_html=True)

    if len(st.session_state.indexed_files) < 2:
        st.info("👆 Please upload at least 2 PDFs to use the comparison feature.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            doc1 = st.selectbox("📄 Document 1", st.session_state.indexed_files, key="doc1")
        with col2:
            doc2 = st.selectbox("📄 Document 2", st.session_state.indexed_files, key="doc2")

        compare_topic = st.text_input("🔍 What aspect do you want to compare?", placeholder="e.g. privacy policy, revenue, marketing strategy")

        if st.button("⚡ Compare Now") and compare_topic and doc1 != doc2:
            with st.spinner("Analyzing both documents..."):
                llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)

                def get_doc_context(doc_name, topic):
                    chunks = st.session_state.doc_chunks.get(doc_name, [])
                    embeddings = st.session_state.embeddings
                    topic_emb = np.array(embeddings.embed_query(topic))
                    scored = []
                    for chunk in chunks:
                        chunk_emb = np.array(embeddings.embed_documents([chunk.page_content])[0])
                        score = np.dot(topic_emb, chunk_emb) / (np.linalg.norm(topic_emb) * np.linalg.norm(chunk_emb))
                        scored.append((score, chunk))
                    scored.sort(key=lambda x: x[0], reverse=True)
                    top_chunks = [c.page_content for _, c in scored[:4]]
                    return "\n\n".join(top_chunks)

                context1 = get_doc_context(doc1, compare_topic)
                context2 = get_doc_context(doc2, compare_topic)

                compare_prompt = PromptTemplate.from_template("""You are an expert document analyst. Compare the two documents below on the topic: {topic}

Document 1 ({doc1}):
{context1}

Document 2 ({doc2}):
{context2}

Provide a detailed comparison covering:
1. How Document 1 addresses this topic
2. How Document 2 addresses this topic
3. Key similarities
4. Key differences
5. Which document covers this topic more thoroughly and why

Comparison:""")

                chain = compare_prompt | llm | StrOutputParser()
                result = chain.invoke({
                    "topic": compare_topic,
                    "doc1": doc1,
                    "doc2": doc2,
                    "context1": context1,
                    "context2": context2
                })

                emb = st.session_state.embeddings
                emb1 = np.array(emb.embed_documents([context1])[0])
                emb2 = np.array(emb.embed_documents([context2])[0])
                similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                similarity_pct = round(float(similarity) * 100, 1)

                m1, m2, m3 = st.columns(3)
                with m1:
                    st.markdown(f'<div class="metric-card"><h3>{similarity_pct}%</h3><p>Document similarity</p></div>', unsafe_allow_html=True)
                with m2:
                    st.markdown(f'<div class="metric-card"><h3>{"🟢" if similarity_pct >= 75 else "🟡" if similarity_pct >= 50 else "🔴"}</h3><p>{"Very similar" if similarity_pct >= 75 else "Somewhat similar" if similarity_pct >= 50 else "Very different"}</p></div>', unsafe_allow_html=True)
                with m3:
                    st.markdown(f'<div class="metric-card"><h3>{compare_topic[:10]}...</h3><p>Topic analyzed</p></div>', unsafe_allow_html=True)

                st.markdown("### 📝 Comparison Result")
                st.markdown(result)

else:
    if not st.session_state.indexed_files:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-card"><h3>📄</h3><p>Upload PDFs in sidebar</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-card"><h3>💬</h3><p>Ask any question</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-card"><h3>🌐</h3><p>Web search included</p></div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about your documents...")

    if question:
        if not groq_key:
            st.warning("⚠️ Please enter your Groq API key in the sidebar.")
        elif st.session_state.vectorstore is None:
            st.warning("⚠️ Please upload at least one PDF first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                    embeddings = st.session_state.embeddings
                    search = DuckDuckGoSearchRun()

                    prompt = PromptTemplate.from_template("""You are a helpful assistant. Use the PDF context and web search results below to answer the question.
Always mention whether the answer came from the PDF, the web, or both.
Always mention page numbers for PDF sources.
If you don't know, say "I couldn't find that in the documents or on the web."

PDF Context: {pdf_context}
Web Search Results: {web_context}
Question: {question}
Answer:""")

                    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
                    docs = retriever.invoke(question)

                    def format_docs(docs):
                        return "\n\n".join(
                            f"[Source: {doc.metadata.get('source', 'unknown')} | Page {doc.metadata.get('page', 0) + 1}]\n{doc.page_content}"
                            for doc in docs
                        )

                    pdf_context = format_docs(docs)

                    with st.spinner("🌐 Searching the web..."):
                        web_context = search.run(question)

                    chain = (
                        {"pdf_context": lambda _: pdf_context,
                         "web_context": lambda _: web_context,
                         "question": RunnablePassthrough()}
                        | prompt
                        | llm
                        | StrOutputParser()
                    )

                    answer = chain.invoke(question)

                    query_embedding = embeddings.embed_query(question)
                    doc_embeddings = embeddings.embed_documents([doc.page_content for doc in docs])
                    query_vec = np.array(query_embedding)
                    scores = []
                    for doc_emb in doc_embeddings:
                        doc_vec = np.array(doc_emb)
                        cosine_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                        scores.append(round(float(cosine_sim) * 100, 1))
                    avg_confidence = round(sum(scores) / len(scores), 1)

                    sources = list(set([
                        f"{doc.metadata.get('source', 'unknown')} p.{doc.metadata.get('page', 0) + 1}"
                        for doc in docs
                    ]))

                    if avg_confidence >= 75:
                        confidence_color = "🟢"
                        confidence_label = "High confidence"
                    elif avg_confidence >= 50:
                        confidence_color = "🟡"
                        confidence_label = "Medium confidence"
                    else:
                        confidence_color = "🔴"
                        confidence_label = "Low confidence"

                    st.markdown(answer)
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        source_tags = " ".join([f'<span class="source-tag">📌 {s}</span>' for s in sources])
                        st.markdown(f'**Sources:** {source_tags}', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'**Confidence:** {confidence_color} {avg_confidence}% — {confidence_label}')

                    full_answer = f"{answer}\n\n📌 *Sources: {', '.join(sources)}*\n\n{confidence_color} *Confidence: {avg_confidence}% — {confidence_label}*"
                    st.session_state.chat_history.append({"role": "assistant", "content": full_answer})