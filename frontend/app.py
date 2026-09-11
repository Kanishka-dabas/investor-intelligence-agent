import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
import pandas as pd
import uuid
from datetime import datetime

from frontend.api_client import stream_chat, get_kpis_from_db, ingest_uploaded_pdf

st.set_page_config(page_title="Investor Intelligence Agent", page_icon="📊", layout="wide")

# ---------- SESSION STATE INIT ----------
if "sessions" not in st.session_state:
    st.session_state.sessions = {}
if "current_session" not in st.session_state:
    new_id = str(uuid.uuid4())[:8]
    st.session_state.sessions[new_id] = {
        "name": "New Chat",
        "messages": [],
    }
    st.session_state.current_session = new_id

# ---------- SIDEBAR ----------
with st.sidebar:
    st.title("📊 Investor Intelligence")

    page = st.radio("Navigate", ["💬 Chat", "📈 KPI Dashboard", "📤 Upload Report"])

    st.divider()
    st.subheader("Chat Sessions")

    if st.button("➕ New Chat"):
        new_id = str(uuid.uuid4())[:8]
        st.session_state.sessions[new_id] = {
            "name": "New Chat",
            "messages": [],
        }
        st.session_state.current_session = new_id
        st.rerun()

    for sid, session_data in reversed(list(st.session_state.sessions.items())):
        is_active = sid == st.session_state.current_session
        label = ("🟢 " if is_active else "") + session_data["name"]
        if st.button(label, key=f"session_{sid}"):
            st.session_state.current_session = sid
            st.rerun()

# ---------- CHAT PAGE ----------
if page == "💬 Chat":
    st.title("Chat with the Investor Intelligence Agent")

    current_session_data = st.session_state.sessions[st.session_state.current_session]
    current_messages = current_session_data["messages"]

    for msg in current_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask about revenue, net income, or live stock prices..."):
        # Auto-name the session from the first message, like ChatGPT does
        if len(current_messages) == 0:
            preview = prompt.strip()[:40]
            new_name = preview + ("..." if len(prompt.strip()) > 40 else "")
            st.session_state.sessions[st.session_state.current_session]["name"] = new_name

        current_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            full_response = ""

            for event_type, content in stream_chat(prompt):
                if event_type == "status":
                    status_placeholder.info(f"🔧 {content}")
                elif event_type == "chunk":
                    full_response += content
                    answer_placeholder.markdown(full_response + "▌")

            status_placeholder.empty()
            answer_placeholder.markdown(full_response)

        current_messages.append({"role": "assistant", "content": full_response})

# ---------- DASHBOARD PAGE ----------
elif page == "📈 KPI Dashboard":
    st.title("Extracted Financial KPIs")

    records = get_kpis_from_db()

    if not records:
        st.info("No KPIs extracted yet. Upload a report first.")
    else:
        df = pd.DataFrame([
            {
                "Company": r.company_name,
                "Period": r.fiscal_period,
                "Revenue": r.revenue,
                "Revenue Unit": r.revenue_unit,
                "Net Income": r.net_income,
                "Net Income Unit": r.net_income_unit,
                "Source": r.source_file,
            }
            for r in records
        ])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Reports Ingested", len(df))
        with col2:
            st.metric("Companies Tracked", df["Company"].nunique())

        st.dataframe(df, use_container_width=True)

        if df["Revenue"].notna().any():
            st.bar_chart(df.set_index("Company")["Revenue"])

# ---------- UPLOAD PAGE ----------
elif page == "📤 Upload Report":
    st.title("Upload a Financial Report")

    uploaded_file = st.file_uploader("Choose a PDF", type="pdf")

    if uploaded_file is not None:
        if st.button("Ingest Report"):
            upload_dir = Path("data/raw_pdfs")
            upload_dir.mkdir(parents=True, exist_ok=True)
            save_path = upload_dir / uploaded_file.name

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner("Processing report — converting, chunking, validating, embedding..."):
                result = ingest_uploaded_pdf(str(save_path), uploaded_file.name)

            if result["status"] == "success":
                st.success(result["message"])
            elif result["status"] == "skipped":
                st.warning(result["message"])
            else:
                st.error(result["message"])