import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.documents import Document
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import tempfile
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
import json

try:
    import pymupdf4llm
    HAS_OCR = True
except ImportError:
    HAS_OCR = False

st.set_page_config(page_title="DocMind AI", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem; border-radius: 12px;
        margin-bottom: 2rem; text-align: center; color: white;
    }
    .main-header h1 { font-size: 2.5rem; margin: 0; }
    .main-header p { opacity: 0.9; margin: 0.5rem 0 0; font-size: 1.1rem; }
    .feature-badge {
        display: inline-block; background: rgba(255,255,255,0.2);
        padding: 4px 12px; border-radius: 20px;
        font-size: 0.8rem; margin: 4px; color: white;
    }
    .metric-card {
        background: #f8f9ff; border: 1px solid #e0e4ff;
        border-radius: 10px; padding: 1rem; text-align: center;
    }
    .metric-card h3 { color: #667eea; font-size: 1.8rem; margin: 0; }
    .metric-card p { color: #666; font-size: 0.85rem; margin: 0.2rem 0 0; }
    .source-tag {
        background: #f0f4ff; border: 1px solid #c7d2fe;
        border-radius: 6px; padding: 2px 8px; font-size: 0.8rem;
        color: #4338ca; margin: 2px; display: inline-block;
    }
    .compare-header {
        background: #f8f9ff; border-left: 4px solid #667eea;
        padding: 1rem; border-radius: 0 8px 8px 0; margin-bottom: 1rem;
    }
    .sidebar-title {
        font-size: 1.1rem; font-weight: 600;
        color: #667eea; margin-bottom: 0.5rem;
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
        <span class="feature-badge">📊 Compare Docs</span>
        <span class="feature-badge">📈 Live Stock Charts</span>
        <span class="feature-badge">💰 Financial Analysis</span>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sidebar-title">⚙️ Configuration</p>', unsafe_allow_html=True)
    groq_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", value=st.secrets.get("GROQ_API_KEY", ""))
    st.markdown("---")
    st.markdown('<p class="sidebar-title">📂 Upload Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("", type="pdf", accept_multiple_files=True, label_visibility="collapsed")
    st.markdown("---")
    st.markdown('<p class="sidebar-title">🎮 Mode</p>', unsafe_allow_html=True)
    mode = st.radio("", ["💬 Chat with PDFs", "📊 Compare Documents", "📈 Financial Analysis"], label_visibility="collapsed")

for key, default in [("chat_history", []), ("chunks", []), ("doc_chunks", {}),
                     ("vectorizer", None), ("tfidf_matrix", None),
                     ("indexed_files", []), ("total_chunks", 0)]:
    if key not in st.session_state:
        st.session_state[key] = default

def retrieve_chunks(question, k=4):
    if not st.session_state.chunks or st.session_state.vectorizer is None:
        return [], 0.0
    question_vec = st.session_state.vectorizer.transform([question])
    scores = cosine_similarity(question_vec, st.session_state.tfidf_matrix)[0]
    top_indices = np.argsort(scores)[::-1][:k]
    top_chunks = [st.session_state.chunks[i] for i in top_indices]
    avg_confidence = round(float(np.mean([scores[i] for i in top_indices])) * 100, 1)
    return top_chunks, avg_confidence

new_files = [f.name for f in uploaded_files] if uploaded_files else []
if new_files != st.session_state.indexed_files and uploaded_files and groq_key:
    progress = st.progress(0, text="Starting indexing...")
    all_chunks = []
    doc_chunks = {}

    for i, uploaded_file in enumerate(uploaded_files):
        progress.progress(i / len(uploaded_files), text=f"Processing {uploaded_file.name}...")
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(uploaded_file.read())
            tmp_path = f.name

        loader = PyMuPDFLoader(tmp_path)
        documents = loader.load()

        total_text = " ".join([doc.page_content for doc in documents]).strip()
        if len(total_text) < 100 and HAS_OCR:
            md_text = pymupdf4llm.to_markdown(tmp_path)
            documents = [Document(page_content=md_text, metadata={"source": uploaded_file.name, "page": 0})]
            st.info(f"🔍 Scanned PDF detected — OCR applied to {uploaded_file.name}")
        elif len(total_text) < 100:
            documents = [Document(page_content="Could not extract text from this PDF.", metadata={"source": uploaded_file.name, "page": 0})]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
        doc_chunks[uploaded_file.name] = chunks
        all_chunks.extend(chunks)

    progress.progress(0.8, text="Building search index...")
    texts = [chunk.page_content for chunk in all_chunks]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    st.session_state.chunks = all_chunks
    st.session_state.doc_chunks = doc_chunks
    st.session_state.vectorizer = vectorizer
    st.session_state.tfidf_matrix = tfidf_matrix
    st.session_state.indexed_files = new_files
    st.session_state.chat_history = []
    st.session_state.total_chunks = len(all_chunks)

    progress.progress(1.0, text="Done!")
    progress.empty()
    st.success(f"✅ Indexed {len(uploaded_files)} PDF(s) — {len(all_chunks)} chunks!")

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
    st.markdown('<div class="compare-header"><h3>📈 Financial Analysis Dashboard</h3><p>Live stock charts + financial metrics from your documents</p></div>', unsafe_allow_html=True)
    tab1, tab2 = st.tabs(["📈 Live Stock Chart", "💰 Financial Metrics from PDF"])

    with tab1:
        col1, col2 = st.columns([2, 1])
        with col1:
            ticker = st.text_input("Enter stock ticker", placeholder="e.g. AAPL, TSLA, GOOGL", value="AAPL")
        with col2:
            period = st.selectbox("Period", ["1mo", "3mo", "6mo", "1y", "2y", "5y"], index=3)

        if st.button("📊 Load Chart"):
            with st.spinner(f"Fetching {ticker}..."):
                try:
                    stock = yf.Ticker(ticker)
                    hist = stock.history(period=period)
                    info = stock.info
                    if hist.empty:
                        st.error("Invalid ticker symbol.")
                    else:
                        current_price = round(hist['Close'].iloc[-1], 2)
                        prev_price = round(hist['Close'].iloc[-2], 2)
                        change_pct = round(((current_price - prev_price) / prev_price) * 100, 2)
                        m1, m2, m3, m4 = st.columns(4)
                        with m1: st.metric("Price", f"${current_price}", f"{change_pct}%")
                        with m2: st.metric("52W High", f"${round(hist['Close'].max(), 2)}")
                        with m3: st.metric("52W Low", f"${round(hist['Close'].min(), 2)}")
                        with m4:
                            mc = info.get('marketCap', 0)
                            st.metric("Market Cap", f"${round(mc/1e9,1)}B" if mc else "N/A")

                        fig = go.Figure()
                        fig.add_trace(go.Candlestick(x=hist.index, open=hist['Open'],
                            high=hist['High'], low=hist['Low'], close=hist['Close'],
                            name=ticker, increasing_line_color='#00C851', decreasing_line_color='#ff4444'))
                        fig.add_trace(go.Scatter(x=hist.index, y=hist['Close'].rolling(20).mean(),
                            name='20-day MA', line=dict(color='#667eea', width=1.5)))
                        fig.update_layout(title=f"{ticker} — {period}", template="plotly_white",
                            height=500, xaxis_rangeslider_visible=False)
                        st.plotly_chart(fig, use_container_width=True)

                        vol_fig = px.bar(hist, x=hist.index, y='Volume', title=f"{ticker} Volume",
                            color='Volume', color_continuous_scale='Blues')
                        vol_fig.update_layout(template="plotly_white", height=250)
                        st.plotly_chart(vol_fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error: {e}")

    with tab2:
        if not st.session_state.chunks:
            st.info("👆 Please upload a financial PDF first")
        else:
            if st.button("🔍 Extract Financial Data"):
                with st.spinner("Extracting..."):
                    docs, _ = retrieve_chunks("revenue net income gross profit financial results", k=6)
                    context = "\n\n".join([doc.page_content for doc in docs])
                    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
                    finance_prompt = PromptTemplate.from_template("""Extract financial metrics from context.
Return ONLY valid JSON (no markdown, no extra text):
{{"company_name": "...", "revenues": [{{"year": "...", "value": ...}}], "net_income": [{{"year": "...", "value": ...}}], "key_insights": ["...", "...", "..."]}}
All monetary values in millions as numbers only.
Context: {context}""")
                    chain = finance_prompt | llm | StrOutputParser()
                    result = chain.invoke({"context": context})
                    try:
                        clean = result.strip().replace("```json", "").replace("```", "").strip()
                        data = json.loads(clean)
                        if data.get("company_name"):
                            st.markdown(f"### {data['company_name']} — Financial Overview")
                        if data.get("revenues"):
                            rev_data = [r for r in data["revenues"] if r.get("value")]
                            if rev_data:
                                fig = go.Figure()
                                fig.add_trace(go.Bar(x=[r["year"] for r in rev_data],
                                    y=[r["value"] for r in rev_data], name="Revenue", marker_color="#667eea"))
                                if data.get("net_income"):
                                    ni = [r for r in data["net_income"] if r.get("value")]
                                    if ni:
                                        fig.add_trace(go.Bar(x=[r["year"] for r in ni],
                                            y=[r["value"] for r in ni], name="Net Income", marker_color="#764ba2"))
                                fig.update_layout(title="Revenue vs Net Income (Millions)",
                                    template="plotly_white", barmode="group", height=400)
                                st.plotly_chart(fig, use_container_width=True)
                        if data.get("key_insights"):
                            st.markdown("### 💡 Key Insights")
                            for insight in data["key_insights"]:
                                if insight: st.markdown(f"- {insight}")
                    except Exception:
                        st.markdown(result)

elif mode == "📊 Compare Documents":
    st.markdown('<div class="compare-header"><h3>📊 Document Comparison</h3></div>', unsafe_allow_html=True)
    if len(st.session_state.indexed_files) < 2:
        st.info("👆 Please upload at least 2 PDFs to compare.")
    else:
        col1, col2 = st.columns(2)
        with col1: doc1 = st.selectbox("📄 Document 1", st.session_state.indexed_files, key="doc1")
        with col2: doc2 = st.selectbox("📄 Document 2", st.session_state.indexed_files, key="doc2")
        compare_topic = st.text_input("🔍 Topic to compare", placeholder="e.g. revenue, privacy, marketing")

        if st.button("⚡ Compare Now") and compare_topic and doc1 != doc2:
            with st.spinner("Analyzing..."):
                def get_doc_context(doc_name, topic):
                    chunks = st.session_state.doc_chunks.get(doc_name, [])
                    if not chunks: return ""
                    texts = [c.page_content for c in chunks]
                    vec = TfidfVectorizer()
                    mat = vec.fit_transform(texts)
                    scores = cosine_similarity(vec.transform([topic]), mat)[0]
                    top = np.argsort(scores)[::-1][:4]
                    return "\n\n".join([chunks[i].page_content for i in top])

                context1 = get_doc_context(doc1, compare_topic)
                context2 = get_doc_context(doc2, compare_topic)
                llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
                compare_prompt = PromptTemplate.from_template("""Compare these documents on: {topic}
Doc 1 ({doc1}): {context1}
Doc 2 ({doc2}): {context2}
Cover: 1) How Doc 1 addresses this 2) How Doc 2 addresses this 3) Similarities 4) Differences 5) Which is more thorough""")
                result = (compare_prompt | llm | StrOutputParser()).invoke(
                    {"topic": compare_topic, "doc1": doc1, "doc2": doc2, "context1": context1, "context2": context2})

                vec = TfidfVectorizer()
                mat = vec.fit_transform([context1, context2])
                sim_pct = round(float(cosine_similarity(mat[0:1], mat[1:2])[0][0]) * 100, 1)
                emoji = "🟢" if sim_pct >= 75 else "🟡" if sim_pct >= 50 else "🔴"
                label = "Very similar" if sim_pct >= 75 else "Somewhat similar" if sim_pct >= 50 else "Very different"

                m1, m2, m3 = st.columns(3)
                with m1: st.markdown(f'<div class="metric-card"><h3>{sim_pct}%</h3><p>Similarity</p></div>', unsafe_allow_html=True)
                with m2: st.markdown(f'<div class="metric-card"><h3>{emoji}</h3><p>{label}</p></div>', unsafe_allow_html=True)
                with m3: st.markdown(f'<div class="metric-card"><h3>{compare_topic[:10]}...</h3><p>Topic</p></div>', unsafe_allow_html=True)
                st.markdown("### 📝 Comparison Result")
                st.markdown(result)

else:
    if not st.session_state.indexed_files:
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="metric-card"><h3>📄</h3><p>Upload PDFs in sidebar</p></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="metric-card"><h3>💬</h3><p>Ask any question</p></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="metric-card"><h3>🌐</h3><p>Web search included</p></div>', unsafe_allow_html=True)

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    question = st.chat_input("Ask anything about your documents...")

    if question:
        if not groq_key:
            st.warning("⚠️ Please enter your Groq API key.")
        elif not st.session_state.chunks:
            st.warning("⚠️ Please upload at least one PDF first.")
        else:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    docs, avg_confidence = retrieve_chunks(question, k=4)
                    try:
                        web_context = DuckDuckGoSearchRun().run(question)
                    except Exception:
                        web_context = "Web search unavailable."

                    pdf_context = "\n\n".join(
                        f"[Source: {doc.metadata.get('source','unknown')} | Page {doc.metadata.get('page',0)+1}]\n{doc.page_content}"
                        for doc in docs)

                    prompt = PromptTemplate.from_template("""You are a helpful assistant. Use PDF context and web results to answer.
Mention whether answer came from PDF, web, or both. Mention page numbers for PDF sources.
PDF Context: {pdf_context}
Web Results: {web_context}
Question: {question}
Answer:""")
                    llm = ChatGroq(api_key=groq_key, model="llama-3.1-8b-instant", temperature=0)
                    answer = (prompt | llm | StrOutputParser()).invoke(
                        {"pdf_context": pdf_context, "web_context": web_context, "question": question})

                    sources = list(set([
                        f"{doc.metadata.get('source','unknown')} p.{doc.metadata.get('page',0)+1}"
                        for doc in docs]))
                    c_color = "🟢" if avg_confidence >= 75 else "🟡" if avg_confidence >= 50 else "🔴"
                    c_label = "High" if avg_confidence >= 75 else "Medium" if avg_confidence >= 50 else "Low"

                    st.markdown(answer)
                    st.markdown("---")
                    col1, col2 = st.columns(2)
                    with col1:
                        tags = " ".join([f'<span class="source-tag">📌 {s}</span>' for s in sources])
                        st.markdown(f'**Sources:** {tags}', unsafe_allow_html=True)
                    with col2:
                        st.markdown(f'**Confidence:** {c_color} {avg_confidence}% — {c_label}')

                    st.session_state.chat_history.append({"role": "assistant",
                        "content": f"{answer}\n\n📌 *Sources: {', '.join(sources)}*\n\n{c_color} *Confidence: {avg_confidence}% — {c_label}*"})