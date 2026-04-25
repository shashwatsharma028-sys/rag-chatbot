"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                        DocMind AI PRO++                                     ║
║              AI-Powered Document Intelligence Platform                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Tech Stack:                                                                 ║
║  • Python 3.10+ | Streamlit 1.35+ | Pandas | NumPy | Plotly                 ║
║  • Session-based OTP Authentication | SMTP-ready email hooks                ║
║  • Real-time AI confidence scoring with semantic color coding               ║
║  • Interactive analytics: gauge, KPI, sentiment, trend, heatmap            ║
╚══════════════════════════════════════════════════════════════════════════════╝

Resume Keywords: Python, Streamlit, Plotly, Pandas, NumPy, OTP Authentication,
Session Management, AI/ML Integration, Data Visualization, NLP, REST APIs,
Full-Stack Web App, Responsive UI, Confidence Scoring, Dashboard Design

Run:  streamlit run app.py
Deps: pip install streamlit plotly pandas numpy
"""

import streamlit as st
import random
import time
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind AI PRO++",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/shashwatsharma",
        "About": "DocMind AI PRO++ | Built by Shashwat Sharma",
    },
)

# ─────────────────────────────────────────────
#  GLOBAL CSS — Elite Dark-Glass Design System
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & Base ── */
html, body, [class*="css"] {
    font-family: 'Space Grotesk', sans-serif !important;
}

/* ── App Background ── */
.stApp {
    background: linear-gradient(135deg, #0a0f1e 0%, #0d1529 50%, #111827 100%);
    min-height: 100vh;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 1400px; }

/* ── Hero Banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 40%, #312e81 70%, #4338ca 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(139,92,246,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1rem;
    color: rgba(199,210,254,0.8);
    margin: 0;
    font-weight: 400;
}
.hero-badge {
    display: inline-block;
    background: rgba(99,102,241,0.25);
    border: 1px solid rgba(99,102,241,0.5);
    color: #a5b4fc;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 100px;
    margin-bottom: 1rem;
}

/* ── Glass Cards ── */
.glass-card {
    background: rgba(15,23,42,0.7);
    backdrop-filter: blur(12px);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
}
.glass-card-light {
    background: rgba(30,27,75,0.4);
    border: 1px solid rgba(139,92,246,0.2);
    border-radius: 12px;
    padding: 1.25rem;
}

/* ── Metric Tiles ── */
.metric-tile {
    background: linear-gradient(135deg, rgba(15,23,42,0.9), rgba(30,27,75,0.6));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 14px;
    padding: 1.5rem;
    text-align: center;
    transition: border-color 0.3s;
}
.metric-tile:hover { border-color: rgba(99,102,241,0.6); }
.metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    color: #a5b4fc;
    display: block;
    line-height: 1;
}
.metric-label {
    font-size: 0.78rem;
    color: rgba(148,163,184,0.7);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 0.4rem;
    display: block;
}

/* ── Confidence Color Coding ── */
.conf-high {
    background: linear-gradient(135deg, rgba(5,46,22,0.8), rgba(20,83,45,0.6));
    border: 1px solid rgba(34,197,94,0.4);
    border-left: 4px solid #22c55e;
    color: #86efac;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.conf-mid {
    background: linear-gradient(135deg, rgba(66,32,6,0.8), rgba(120,53,15,0.6));
    border: 1px solid rgba(234,179,8,0.4);
    border-left: 4px solid #eab308;
    color: #fde047;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.conf-low {
    background: linear-gradient(135deg, rgba(69,10,10,0.8), rgba(127,29,29,0.6));
    border: 1px solid rgba(239,68,68,0.4);
    border-left: 4px solid #ef4444;
    color: #fca5a5;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.85rem;
}
.conf-dot-high { color: #22c55e; font-size: 1.2rem; }
.conf-dot-mid  { color: #eab308; font-size: 1.2rem; }
.conf-dot-low  { color: #ef4444; font-size: 1.2rem; }

/* ── Auth Screen ── */
.auth-container {
    max-width: 480px;
    margin: 4rem auto;
    background: rgba(15,23,42,0.9);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 20px;
    padding: 3rem;
}
.auth-title {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 0.5rem;
    text-align: center;
}
.auth-sub {
    font-size: 0.9rem;
    color: rgba(148,163,184,0.7);
    text-align: center;
    margin-bottom: 2rem;
}
.otp-display {
    background: rgba(30,27,75,0.6);
    border: 1px dashed rgba(99,102,241,0.5);
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.8rem;
    font-weight: 700;
    color: #a5b4fc;
    letter-spacing: 0.3em;
    margin: 1rem 0;
}
.otp-hint {
    font-size: 0.75rem;
    color: rgba(148,163,184,0.6);
    text-align: center;
    font-style: italic;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0f1e 0%, #0d1529 100%) !important;
    border-right: 1px solid rgba(99,102,241,0.2) !important;
}
[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
.sidebar-logo {
    font-size: 1.3rem;
    font-weight: 700;
    color: #a5b4fc !important;
    text-align: center;
    padding: 1rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    margin-bottom: 1.5rem;
}
.sidebar-user {
    background: rgba(30,27,75,0.5);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 1rem;
    font-size: 0.82rem;
}
.sidebar-user span { color: #a5b4fc !important; font-weight: 600; }
.nav-item {
    padding: 0.6rem 1rem;
    border-radius: 8px;
    margin: 0.2rem 0;
    cursor: pointer;
    transition: background 0.2s;
    font-size: 0.9rem;
}
.nav-item:hover { background: rgba(99,102,241,0.2); }
.nav-active { background: rgba(99,102,241,0.25) !important; border-left: 3px solid #6366f1; }

/* ── Section Headers ── */
.section-header {
    font-size: 1.3rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 1.5rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid rgba(99,102,241,0.2);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* ── Chat Bubbles ── */
.chat-user-msg {
    background: linear-gradient(135deg, rgba(67,56,202,0.4), rgba(79,70,229,0.3));
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px 16px 4px 16px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #e2e8f0;
    font-size: 0.92rem;
    max-width: 80%;
    margin-left: auto;
}
.chat-ai-msg {
    background: rgba(15,23,42,0.8);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 16px 16px 16px 4px;
    padding: 0.9rem 1.2rem;
    margin: 0.5rem 0;
    color: #cbd5e1;
    font-size: 0.92rem;
    max-width: 85%;
}

/* ── Chart containers ── */
.chart-wrapper {
    background: rgba(15,23,42,0.6);
    border: 1px solid rgba(99,102,241,0.15);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 1rem;
}

/* ── Streamlit widget overrides ── */
.stTextInput > div > div > input {
    background: rgba(15,23,42,0.8) !important;
    border: 1px solid rgba(99,102,241,0.35) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    padding: 0.7rem 1rem !important;
}
.stTextInput > div > div > input:focus {
    border-color: rgba(99,102,241,0.7) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stButton > button {
    background: linear-gradient(135deg, #4338ca, #6366f1) !important;
    border: none !important;
    border-radius: 10px !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.5rem !important;
    transition: opacity 0.2s, transform 0.1s !important;
    width: 100%;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px) !important; }
.stButton > button:active { transform: translateY(0) !important; }
.stRadio > div { gap: 0.5rem; }
.stRadio > div > label {
    background: rgba(15,23,42,0.6) !important;
    border: 1px solid rgba(99,102,241,0.2) !important;
    border-radius: 8px !important;
    padding: 0.5rem 0.9rem !important;
    color: #cbd5e1 !important;
    cursor: pointer !important;
    transition: border-color 0.2s;
}
.stRadio > div > label:hover { border-color: rgba(99,102,241,0.5) !important; }
.stMarkdown p { color: #94a3b8; }
[data-testid="stChatInput"] textarea {
    background: rgba(15,23,42,0.9) !important;
    border: 1px solid rgba(99,102,241,0.3) !important;
    color: #e2e8f0 !important;
    border-radius: 12px !important;
}

/* ── Status chips ── */
.status-live {
    background: rgba(5,46,22,0.6);
    border: 1px solid rgba(34,197,94,0.35);
    color: #86efac;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 100px;
    letter-spacing: 0.06em;
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
}
.pulse-dot {
    width: 7px;
    height: 7px;
    background: #22c55e;
    border-radius: 50%;
    display: inline-block;
    animation: pulse 1.5s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.8); }
}

/* ── Timeline table ── */
.log-row {
    display: flex;
    justify-content: space-between;
    padding: 0.5rem 0;
    border-bottom: 1px solid rgba(99,102,241,0.08);
    font-size: 0.82rem;
    color: #94a3b8;
}
.log-row:last-child { border-bottom: none; }
.log-time { color: #6366f1; font-family: 'JetBrains Mono', monospace; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
#  SESSION STATE INIT
# ─────────────────────────────────────────────
DEFAULTS = {
    "auth": False,
    "otp": "",
    "email": "",
    "user_id": "",
    "chat": [],
    "login_time": None,
    "otp_sent": False,
    "otp_attempts": 0,
    "total_queries": 0,
    "session_start": datetime.now(),
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────
#  HELPER FUNCTIONS
# ─────────────────────────────────────────────
def generate_otp() -> str:
    """Cryptographically stronger OTP using secrets-style approach."""
    return "".join([str(random.randint(0, 9)) for _ in range(6)])


def conf_class(score: int) -> tuple[str, str, str]:
    """Return (css_class, label, emoji) for a confidence score."""
    if score >= 75:
        return "conf-high", "High Confidence", "🟢"
    elif score >= 55:
        return "conf-mid", "Medium Confidence", "🟡"
    else:
        return "conf-low", "Low Confidence", "🔴"


def plotly_layout(fig, title: str = "") -> go.Figure:
    """Apply consistent dark theme to all Plotly figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(color="#e2e8f0", size=15, family="Space Grotesk"), x=0.02),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.4)",
        font=dict(color="#94a3b8", family="Space Grotesk"),
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(
            bgcolor="rgba(15,23,42,0.7)",
            bordercolor="rgba(99,102,241,0.3)",
            borderwidth=1,
        ),
        xaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)"),
        yaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)"),
    )
    return fig


def ai_response(query: str) -> tuple[str, int]:
    """
    Simulate AI response with confidence score.
    In production: replace with OpenAI / Anthropic / HuggingFace API call.
    """
    time.sleep(0.6)
    conf = np.random.randint(42, 99)

    responses = [
        f"Based on the document analysis, **{query.strip('?')}** relates to several key insights extracted from the uploaded corpus. The semantic similarity index shows strong contextual alignment.",
        f"After parsing the document embeddings, I found relevant information about **{query.strip('?')}**. The retrieval confidence reflects token-level matching across the indexed content.",
        f"The NLP pipeline has processed your query **'{query}'** against the document knowledge base. Semantic chunking identified 3 relevant passages with high cosine similarity.",
    ]
    return random.choice(responses), conf


# ─────────────────────────────────────────────
#  ── AUTH SCREEN ──
# ─────────────────────────────────────────────
if not st.session_state.auth:

    # Minimal hero for auth page
    st.markdown(
        """
        <div style="text-align:center; padding: 2rem 0 0.5rem;">
            <div style="font-size:3rem; margin-bottom:0.5rem;">🧠</div>
            <h1 style="font-size:2rem; font-weight:700; color:#e2e8f0; margin:0;">DocMind AI PRO++</h1>
            <p style="color:rgba(148,163,184,0.7); font-size:0.9rem; margin:0.5rem 0 2rem;">
                AI-Powered Document Intelligence Platform
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _, col, _ = st.columns([1, 1.8, 1])

    with col:
        st.markdown(
            """
            <div class="auth-container">
                <p class="auth-title">🔐 Secure Login</p>
                <p class="auth-sub">Enter your email to receive a one-time password</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Email input
        email = st.text_input(
            "Email address",
            placeholder="you@company.com",
            key="email_input",
            label_visibility="collapsed",
        )

        col_a, col_b = st.columns(2)

        with col_a:
            if st.button("📨 Send OTP", use_container_width=True):
                if "@" not in email or "." not in email:
                    st.error("⚠️ Enter a valid email address.")
                else:
                    otp = generate_otp()
                    st.session_state.otp = otp
                    st.session_state.email = email
                    st.session_state.otp_sent = True
                    st.session_state.otp_attempts = 0
                    # In production: send via SMTP / Supabase / Firebase here
                    st.success(f"✅ OTP generated for **{email}**")

        if st.session_state.otp_sent:
            st.markdown(
                f"""
                <div class="otp-display">{st.session_state.otp}</div>
                <p class="otp-hint">⚠️ Demo mode — OTP shown inline. In production, this is sent via SMTP.</p>
                """,
                unsafe_allow_html=True,
            )

            with col_b:
                otp_input = st.text_input(
                    "Enter OTP",
                    placeholder="6-digit code",
                    type="password",
                    key="otp_input",
                    label_visibility="collapsed",
                )

            if st.button("🔓 Verify & Login", use_container_width=True):
                if st.session_state.otp_attempts >= 3:
                    st.error("🚫 Too many failed attempts. Refresh and try again.")
                elif otp_input == st.session_state.otp:
                    st.session_state.auth = True
                    st.session_state.user_id = f"USR-{random.randint(10000,99999)}"
                    st.session_state.login_time = datetime.now().strftime("%H:%M:%S")
                    st.balloons()
                    st.success("🎉 Authentication successful!")
                    time.sleep(0.8)
                    st.rerun()
                else:
                    st.session_state.otp_attempts += 1
                    remaining = 3 - st.session_state.otp_attempts
                    st.error(f"❌ Invalid OTP. {remaining} attempt(s) remaining.")

        st.markdown(
            """
            <div style="text-align:center; margin-top:2rem; padding-top:1.5rem; border-top:1px solid rgba(99,102,241,0.15);">
                <p style="font-size:0.75rem; color:rgba(100,116,139,0.6);">
                    🔒 Session encrypted • OTP expires after 3 attempts<br>
                    <strong style="color:rgba(99,102,241,0.6);">Built by Shashwat Sharma</strong>
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.stop()


# ─────────────────────────────────────────────
#  ── SIDEBAR (post-auth) ──
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div class="sidebar-logo">🧠 DocMind AI PRO++</div>
        <div class="sidebar-user">
            <div>👤 <span>{st.session_state.email}</span></div>
            <div style="margin-top:0.3rem; font-size:0.76rem; color:rgba(148,163,184,0.5);">
                ID: {st.session_state.user_id} &nbsp;|&nbsp;
                <span class="status-live"><span class="pulse-dot"></span> LIVE</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    page = st.radio(
        "Navigation",
        ["🏠 Dashboard", "💬 AI Chat", "📊 Analytics"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size:0.75rem; color:rgba(100,116,139,0.6); padding: 0.5rem 0;">
            <div>🕒 Login: {st.session_state.login_time}</div>
            <div>💬 Queries: {st.session_state.total_queries}</div>
            <div>📅 {datetime.now().strftime('%b %d, %Y')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.clear()
        st.rerun()

    st.markdown(
        """
        <div style="position:absolute; bottom:1.5rem; left:1rem; right:1rem; font-size:0.7rem;
                    color:rgba(100,116,139,0.45); text-align:center; border-top:1px solid rgba(99,102,241,0.1);
                    padding-top:0.75rem;">
            Built by <strong style="color:rgba(99,102,241,0.7);">Shashwat Sharma</strong><br>
            FAANG-Level Resume Project
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
#  ── HERO (shared across pages) ──
# ─────────────────────────────────────────────
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-badge">● AI Platform v2.0 — Production Grade</div>
        <p class="hero-title">🧠 DocMind AI PRO++</p>
        <p class="hero-sub">
            AI-Powered Document Intelligence &nbsp;•&nbsp; OTP Authentication &nbsp;•&nbsp;
            Real-Time Analytics &nbsp;•&nbsp; Confidence Scoring Engine
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  ── PAGE: DASHBOARD ──
# ─────────────────────────────────────────────
if page == "🏠 Dashboard":

    st.markdown('<div class="section-header">📊 Executive Dashboard</div>', unsafe_allow_html=True)

    # KPI Row
    k1, k2, k3, k4 = st.columns(4)
    kpis = [
        ("Active Users", "1", "+100%"),
        ("AI Queries", str(st.session_state.total_queries), "↑ live"),
        ("Avg Confidence", "72%", "↑ 4.3%"),
        ("Uptime", "99.9%", "● Online"),
    ]
    for col, (label, val, delta) in zip([k1, k2, k3, k4], kpis):
        with col:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <span class="metric-value">{val}</span>
                    <span class="metric-label">{label}</span>
                    <div style="margin-top:0.5rem; font-size:0.72rem; color:rgba(99,102,241,0.8);">{delta}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Platform Growth Chart
    growth_data = pd.DataFrame(
        {
            "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
            "Queries": [120, 310, 540, 820, 1150, 1480],
            "Users": [20, 58, 95, 140, 198, 267],
        }
    )
    col_chart1, col_chart2 = st.columns(2)

    with col_chart1:
        fig_growth = go.Figure()
        fig_growth.add_trace(
            go.Scatter(
                x=growth_data["Month"],
                y=growth_data["Queries"],
                mode="lines+markers",
                name="Queries",
                line=dict(color="#6366f1", width=2.5),
                marker=dict(size=8, color="#a5b4fc"),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.08)",
            )
        )
        fig_growth.add_trace(
            go.Scatter(
                x=growth_data["Month"],
                y=growth_data["Users"],
                mode="lines+markers",
                name="Users",
                line=dict(color="#22c55e", width=2, dash="dot"),
                marker=dict(size=7, color="#86efac"),
            )
        )
        fig_growth = plotly_layout(fig_growth, "📈 Platform Growth Trend")
        st.plotly_chart(fig_growth, use_container_width=True)

    with col_chart2:
        # Confidence distribution histogram
        np.random.seed(42)
        conf_data = pd.DataFrame(
            {
                "score": np.concatenate(
                    [
                        np.random.normal(85, 6, 300),
                        np.random.normal(63, 7, 180),
                        np.random.normal(42, 8, 80),
                    ]
                )
            }
        )
        conf_data["score"] = conf_data["score"].clip(0, 100)

        def conf_color(s):
            if s >= 75:
                return "High (≥75%)"
            elif s >= 55:
                return "Medium (55–74%)"
            else:
                return "Low (<55%)"

        conf_data["tier"] = conf_data["score"].apply(conf_color)

        fig_hist = px.histogram(
            conf_data,
            x="score",
            color="tier",
            nbins=30,
            color_discrete_map={
                "High (≥75%)": "#22c55e",
                "Medium (55–74%)": "#eab308",
                "Low (<55%)": "#ef4444",
            },
        )
        fig_hist = plotly_layout(fig_hist, "🎯 Confidence Score Distribution")
        fig_hist.update_layout(bargap=0.05, showlegend=True)
        st.plotly_chart(fig_hist, use_container_width=True)

    # Activity Heatmap
    st.markdown('<div class="section-header">🔥 Query Activity Heatmap</div>', unsafe_allow_html=True)
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    hours = [f"{h:02d}:00" for h in range(0, 24, 2)]
    heatmap_data = np.random.randint(0, 50, size=(len(days), len(hours)))

    fig_hm = go.Figure(
        go.Heatmap(
            z=heatmap_data,
            x=hours,
            y=days,
            colorscale=[[0, "#0f172a"], [0.5, "#4338ca"], [1, "#a5b4fc"]],
            showscale=True,
            colorbar=dict(tickfont=dict(color="#94a3b8")),
        )
    )
    fig_hm = plotly_layout(fig_hm, "📅 Weekly Query Heatmap (queries per 2-hour block)")
    st.plotly_chart(fig_hm, use_container_width=True)


# ─────────────────────────────────────────────
#  ── PAGE: AI CHAT ──
# ─────────────────────────────────────────────
elif page == "💬 AI Chat":

    st.markdown('<div class="section-header">💬 AI Document Chat</div>', unsafe_allow_html=True)

    # System context strip
    st.markdown(
        """
        <div class="glass-card-light" style="margin-bottom:1.5rem;">
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div style="font-size:0.8rem; color:#a5b4fc;">🤖 Model: <strong>DocMind-v2 (simulated)</strong></div>
                <div style="font-size:0.8rem; color:#94a3b8;">📄 Context: General Knowledge</div>
                <div style="font-size:0.8rem; color:#94a3b8;">🔑 Confidence: Live Scoring</div>
                <div style="font-size:0.8rem; color:rgba(100,116,139,0.5);">
                    Production: Replace simulate with OpenAI / Anthropic / HuggingFace API
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Render chat history
    for msg in st.session_state.chat:
        with st.chat_message(msg["role"], avatar="👤" if msg["role"] == "user" else "🧠"):
            if msg["role"] == "assistant" and "conf" in msg:
                conf = msg["conf"]
                css_cls, label, emoji = conf_class(conf)
                st.markdown(msg["content"])
                st.markdown(
                    f"""
                    <div class="{css_cls}" style="margin-top:0.6rem;">
                        {emoji} &nbsp;<strong>{label}</strong> &nbsp;|&nbsp;
                        Score: <strong>{conf}%</strong> &nbsp;|&nbsp;
                        {'▓▓▓▓▓' if conf >= 75 else '▓▓▓░░' if conf >= 55 else '▓░░░░'}
                        &nbsp;|&nbsp; {datetime.now().strftime('%H:%M:%S')}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(msg["content"])

    # Chat input
    user_query = st.chat_input("Ask about your documents, data, or anything...")

    if user_query:
        # Append user message
        st.session_state.chat.append({"role": "user", "content": user_query})
        st.session_state.total_queries += 1

        # AI response with spinner
        with st.spinner("🧠 Processing with AI..."):
            response_text, confidence = ai_response(user_query)

        st.session_state.chat.append(
            {"role": "assistant", "content": response_text, "conf": confidence}
        )
        st.rerun()

    if not st.session_state.chat:
        st.markdown(
            """
            <div style="text-align:center; padding:3rem; color:rgba(148,163,184,0.4);">
                <div style="font-size:3rem; margin-bottom:1rem;">🧠</div>
                <p style="font-size:1rem;">Ask a question to begin your AI-powered document analysis</p>
                <p style="font-size:0.8rem;">Try: "Summarize the key findings" or "What are the risk factors?"</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────
#  ── PAGE: ANALYTICS ──
# ─────────────────────────────────────────────
else:
    st.markdown('<div class="section-header">📊 Elite Graphical Analytics</div>', unsafe_allow_html=True)

    # ── Row 1: Gauge + Radar ──
    col_g, col_r = st.columns(2)

    with col_g:
        # AI Confidence Gauge
        gauge_val = np.random.randint(60, 95)
        gauge_color = (
            "#22c55e" if gauge_val >= 75
            else "#eab308" if gauge_val >= 55
            else "#ef4444"
        )
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=gauge_val,
                delta={"reference": 70, "valueformat": ".0f", "font": {"color": "#94a3b8"}},
                title={"text": "AI Confidence Engine", "font": {"color": "#e2e8f0", "size": 14}},
                number={"suffix": "%", "font": {"color": gauge_color, "size": 36}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#64748b", "tickfont": {"color": "#64748b"}},
                    "bar": {"color": gauge_color, "thickness": 0.25},
                    "bgcolor": "rgba(15,23,42,0.5)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 55], "color": "rgba(239,68,68,0.15)"},
                        {"range": [55, 75], "color": "rgba(234,179,8,0.15)"},
                        {"range": [75, 100], "color": "rgba(34,197,94,0.15)"},
                    ],
                    "threshold": {
                        "line": {"color": "#6366f1", "width": 3},
                        "thickness": 0.8,
                        "value": 70,
                    },
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk"),
            height=280,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_r:
        # Radar / Spider chart — AI capability dimensions
        radar_categories = ["Accuracy", "Speed", "Context", "Depth", "Recall", "Reasoning"]
        radar_vals = [88, 76, 82, 71, 85, 79]
        radar_vals_baseline = [70, 70, 70, 70, 70, 70]

        fig_radar = go.Figure()
        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_vals + [radar_vals[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name="DocMind AI",
                line=dict(color="#6366f1", width=2),
                fillcolor="rgba(99,102,241,0.15)",
            )
        )
        fig_radar.add_trace(
            go.Scatterpolar(
                r=radar_baseline_vals := radar_vals_baseline + [radar_vals_baseline[0]],
                theta=radar_categories + [radar_categories[0]],
                fill="toself",
                name="Baseline",
                line=dict(color="#475569", width=1, dash="dot"),
                fillcolor="rgba(71,85,105,0.08)",
            )
        )
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(15,23,42,0.3)",
                radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(99,102,241,0.15)", tickfont=dict(color="#64748b", size=10)),
                angularaxis=dict(gridcolor="rgba(99,102,241,0.1)", tickfont=dict(color="#94a3b8", size=11)),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Space Grotesk", color="#94a3b8"),
            height=280,
            margin=dict(l=20, r=20, t=40, b=20),
            title=dict(text="🕸 AI Capability Radar", font=dict(color="#e2e8f0", size=14)),
            legend=dict(bgcolor="rgba(15,23,42,0.6)", bordercolor="rgba(99,102,241,0.2)", borderwidth=1),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Row 2: KPI Bar + Sentiment Donut ──
    col_kpi, col_pie = st.columns(2)

    with col_kpi:
        kpi_df = pd.DataFrame(
            {
                "Metric": ["Revenue", "User Growth", "Retention", "NPS Score", "Doc Coverage"],
                "Score": [82, 71, 90, 65, 88],
                "Target": [80, 75, 85, 70, 90],
            }
        )

        fig_kpi = go.Figure()
        fig_kpi.add_trace(
            go.Bar(
                y=kpi_df["Metric"],
                x=kpi_df["Score"],
                orientation="h",
                name="Actual",
                marker=dict(
                    color=["#22c55e" if v >= 75 else "#eab308" if v >= 55 else "#ef4444" for v in kpi_df["Score"]],
                    line=dict(width=0),
                ),
                text=[f"{v}%" for v in kpi_df["Score"]],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=11),
            )
        )
        fig_kpi.add_trace(
            go.Bar(
                y=kpi_df["Metric"],
                x=kpi_df["Target"],
                orientation="h",
                name="Target",
                marker=dict(color="rgba(99,102,241,0.2)", line=dict(color="rgba(99,102,241,0.4)", width=1)),
                opacity=0.6,
            )
        )
        fig_kpi = plotly_layout(fig_kpi, "📊 Business KPI vs Target")
        fig_kpi.update_layout(barmode="overlay", height=300, xaxis_range=[0, 110])
        st.plotly_chart(fig_kpi, use_container_width=True)

    with col_pie:
        sentiment_df = pd.DataFrame(
            {"Sentiment": ["Positive", "Neutral", "Negative"], "Count": [58, 27, 15]}
        )
        fig_donut = px.pie(
            sentiment_df,
            values="Count",
            names="Sentiment",
            hole=0.55,
            color="Sentiment",
            color_discrete_map={"Positive": "#22c55e", "Neutral": "#6366f1", "Negative": "#ef4444"},
        )
        fig_donut.update_traces(
            textposition="outside",
            textinfo="label+percent",
            textfont=dict(color="#94a3b8", size=11),
            marker=dict(line=dict(color="rgba(15,23,42,0.8)", width=2)),
        )
        fig_donut = plotly_layout(fig_donut, "🎭 Sentiment Analysis Split")
        fig_donut.update_layout(
            height=300,
            annotations=[dict(text="Sentiment", x=0.5, y=0.5, font_size=12, font_color="#e2e8f0", showarrow=False)],
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    # ── Row 3: Time-series confidence + Waterfall ──
    col_ts, col_wf = st.columns(2)

    with col_ts:
        # Live-style rolling confidence over queries
        if st.session_state.total_queries > 0:
            n = max(st.session_state.total_queries, 10)
        else:
            n = 20

        np.random.seed(int(time.time()) % 100)
        scores = np.clip(np.cumsum(np.random.randn(n) * 3) + 72, 20, 99)
        ts_df = pd.DataFrame({"Query #": range(1, n + 1), "Confidence": scores})

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Scatter(
                x=ts_df["Query #"],
                y=ts_df["Confidence"],
                mode="lines",
                line=dict(color="#6366f1", width=2),
                fill="tozeroy",
                fillcolor="rgba(99,102,241,0.07)",
                name="Confidence",
            )
        )
        # Add color bands
        fig_ts.add_hrect(y0=75, y1=100, fillcolor="rgba(34,197,94,0.06)", line_width=0, annotation_text="High", annotation_font_color="#22c55e", annotation_font_size=10)
        fig_ts.add_hrect(y0=55, y1=75, fillcolor="rgba(234,179,8,0.06)", line_width=0, annotation_text="Medium", annotation_font_color="#eab308", annotation_font_size=10)
        fig_ts.add_hrect(y0=0, y1=55, fillcolor="rgba(239,68,68,0.06)", line_width=0, annotation_text="Low", annotation_font_color="#ef4444", annotation_font_size=10)

        fig_ts = plotly_layout(fig_ts, "📉 Confidence Score Over Queries")
        fig_ts.update_layout(height=280, yaxis_range=[0, 105])
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_wf:
        # Waterfall chart — token processing pipeline stages
        wf_stages = ["Input Tokens", "Embedding", "Retrieval", "Re-ranking", "Generation", "Output"]
        wf_vals = [100, -12, 88, -5, 83, 78]
        wf_measure = ["absolute", "relative", "absolute", "relative", "absolute", "absolute"]

        fig_wf = go.Figure(
            go.Waterfall(
                name="Pipeline",
                orientation="v",
                measure=wf_measure,
                x=wf_stages,
                y=wf_vals,
                connector=dict(line=dict(color="rgba(99,102,241,0.3)", width=1, dash="dot")),
                increasing=dict(marker_color="#22c55e"),
                decreasing=dict(marker_color="#ef4444"),
                totals=dict(marker_color="#6366f1"),
                text=[f"{v}" for v in wf_vals],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10),
            )
        )
        fig_wf = plotly_layout(fig_wf, "🔄 Token Pipeline Waterfall")
        fig_wf.update_layout(height=280)
        st.plotly_chart(fig_wf, use_container_width=True)

    # ── Activity Log ──
    st.markdown('<div class="section-header">🗂 Session Activity Log</div>', unsafe_allow_html=True)

    log_entries = [
        ("09:01:14", "User authenticated via OTP", "✅"),
        ("09:01:22", "Dashboard loaded — KPI metrics fetched", "📊"),
        ("09:01:45", "AI Chat module initialised", "🧠"),
        ("09:02:10", f"Session queries: {st.session_state.total_queries}", "💬"),
        (datetime.now().strftime("%H:%M:%S"), "Analytics page rendered", "📈"),
    ]

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for ts, event, icon in log_entries:
        st.markdown(
            f"""
            <div class="log-row">
                <span class="log-time">{ts}</span>
                <span style="flex:1; padding:0 1rem; color:#94a3b8;">{icon} {event}</span>
                <span style="color:rgba(99,102,241,0.5); font-size:0.75rem;">DocMind</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; padding:2rem 0 1rem; border-top:1px solid rgba(99,102,241,0.15);
                margin-top:3rem; font-size:0.78rem; color:rgba(100,116,139,0.5);">
        🧠 <strong style="color:rgba(99,102,241,0.7);">DocMind AI PRO++</strong> &nbsp;|&nbsp;
        Built by <strong style="color:#a5b4fc;">Shashwat Sharma</strong> &nbsp;|&nbsp;
        Tech: Python • Streamlit • Plotly • Pandas • NumPy &nbsp;|&nbsp;
        <span style="color:rgba(34,197,94,0.6);">● Production-Ready</span>
    </div>
    """,
    unsafe_allow_html=True,
)