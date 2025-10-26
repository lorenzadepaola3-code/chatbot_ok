

import uuid
import streamlit as st
from core.app_logger import get_logger
from core.chatbot import (
    generate_ecb_speech_response,
    generate_conversation_title,
)
from tools.sentiment import sentiment_facets_summary, USE_SENTIMENT
import json
import os

logger = get_logger(__name__)
logger.info("Monetary Policy Learning Assistant application started")

# Session state initialization
if "conversations" not in st.session_state:
    st.session_state.conversations = {}
if "conversation_titles" not in st.session_state:
    st.session_state.conversation_titles = {}
if "audience_level" not in st.session_state:
    st.session_state.audience_level = "general"
if "last_user_input" not in st.session_state:
    st.session_state.last_user_input = ""

# Audience mapping (students removed)
audience_map = {
    "general": "general",
    "professional": "professional",
    "basic": "general",  # legacy safety
}
mapped_level = audience_map.get(st.session_state.audience_level, "general")

# Create default conversation
if not st.session_state.conversations:
    default_id = str(uuid.uuid4())
    st.session_state.conversations[default_id] = []
    st.session_state.current_conversation_id = default_id
    st.session_state.conversation_titles[default_id] = ""

# Color scheme
PRIMARY_BLUE = "#004494"
ACCENT_YELLOW = "#FFCC00"
BG_WHITE = "#FFFFFF"
LIGHT_PANEL = "#F8F9FA"

# Styling (students removed, simpler UI, no middle blue bar)
st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BG_WHITE}; }}
    .main-title {{ color: {PRIMARY_BLUE}; font-size: 2.4em; font-weight: 600; text-align: center; margin-bottom: 0.2em; }}
    .subtitle {{ color: {PRIMARY_BLUE}; font-size: 1.05em; text-align: center; margin-bottom: 1.2em; }}
    .audience-badge {{ display:inline-block; padding:4px 10px; border-radius:14px; font-size:0.75em; font-weight:600; }}
    .general-badge {{ background-color:{ACCENT_YELLOW}; color:{PRIMARY_BLUE}; }}
    .professional-badge {{ background-color:#6C757D; color:#FFFFFF; }}
    .assistant-response {{ background-color:{BG_WHITE}; color:{PRIMARY_BLUE}; padding:1.1em 1.3em; border-radius:10px; margin:0.8em 0; border:2px solid {PRIMARY_BLUE}; line-height:1.55; }}
    .general-response {{ background-color:#FFF9E6; border-color:{ACCENT_YELLOW}; }}
    .professional-response {{ background-color:#F1F3F5; border-color:#6C757D; font-size:0.96em; }}
    .info-box {{
        background:#EEF5FF;
        border-left:5px solid {PRIMARY_BLUE};
        padding:12px 16px;
        margin-bottom:16px;
        border-radius:6px;
        font-size:0.9em;
        line-height:1.4;
    }}
    .examples {{ font-family:monospace; font-size:0.85em; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# Title
st.markdown(
    '<h1 class="main-title">Monetary Policy Learning Assistant</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtitle">Ask about ECB speeches, policy stances and topics</p>',
    unsafe_allow_html=True,
)

# Instructions / guidance panel
st.markdown(
    """
    <div class="info-box">
    <strong>Tips:</strong><br>
    Better to write the full speaker name (e.g. <em>Christine Lagarde</em>).<br>
    Start a new conversation for each distinct question to keep answers focused.<br>
    Use clear comparison wording for contrasts (e.g. "compare", "difference", "vs").<br><br>
    <strong>Example queries:</strong>
    <div class="examples">
    1. What did Piero Cipollone say about the digital euro in 2025?<br>
    2. Compare what Isabel Schnabel and Christine Lagarde say about inflation in 2025.<br>
    3. What did Christine Lagarde say about interest rates in 2024?<br>
    4. What did Mickey Mouse say about inflation? (tests handling of unknown speaker)
    </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Audience selector (students removed)
c1, c2 = st.columns(2)
with c1:
    if st.button(" General Public ", key="general_level"):
        st.session_state.audience_level = "general"
with c2:
    if st.button(" Professional ", key="professional_level"):
        st.session_state.audience_level = "professional"

badge_class = "general-badge" if mapped_level == "general" else "professional-badge"
badge_label = (
    "General audience" if mapped_level == "general" else "Professional audience"
)
st.markdown(
    f'<span class="audience-badge {badge_class}">Current Level: {badge_label}</span>',
    unsafe_allow_html=True,
)

# Sidebar: conversation management
with st.sidebar:
    st.markdown("### Conversations")
    if st.button("Start New Conversation", key="new_conversation"):
        new_id = str(uuid.uuid4())
        st.session_state.conversations[new_id] = []
        st.session_state.current_conversation_id = new_id
        st.session_state.conversation_titles[new_id] = ""
    for conv_id, conv_history in st.session_state.conversations.items():
        title = st.session_state.conversation_titles.get(conv_id, "") or "Untitled"
        if st.button(title, key=f"btn_{conv_id}"):
            st.session_state.current_conversation_id = conv_id

# Load metadata once (for filters)
try:
    with open(
        "processed_ecb_data/ecb_speeches_metadata.json", "r", encoding="utf-8"
    ) as f:
        metadata = json.load(f)
except Exception:
    metadata = []

all_speakers = sorted(list({s for m in metadata for s in m.get("speakers", []) if s}))
years = sorted(
    {m["date"][:4] for m in metadata if m.get("date", "")[:4].isdigit()}, reverse=True
)
topics = [
    "inflation",
    "interest rates",
    "monetary policy",
    "digital euro",
    "growth",
    "quantitative easing",
]

fc1, fc2, fc3 = st.columns(3)
with fc1:
    selected_speaker = st.selectbox("Filter Speaker", ["Any"] + all_speakers)
with fc2:
    selected_year = st.selectbox("Filter Year", ["Any"] + years)
with fc3:
    selected_topic = st.selectbox("Topic Keyword", ["Any"] + topics)

# Input form (Enter submits; field clears automatically)
placeholders = {
    "general": "Ask in simple words (e.g. What did Christine Lagarde say about inflation in 2024?)",
    "professional": "Ask a focused policy question (e.g. Compare Lagarde and Schnabel on inflation 2025)",
}
with st.form("query_form", clear_on_submit=True):
    user_input = st.text_input(
        "Your question:",
        key="user_input",
        placeholder=placeholders[mapped_level],
    )
    submitted = st.form_submit_button("Ask Question")

if submitted and user_input:
    try:
        if st.session_state.current_conversation_id is None:
            st.warning("Start or select a conversation first.")
        else:
            conv_id = st.session_state.current_conversation_id
            conv_history = st.session_state.conversations[conv_id]

            if len(conv_history) == 0:
                title = generate_conversation_title(user_input)
                st.session_state.conversation_titles[conv_id] = title

            conv_history.append({"role": "user", "content": user_input})

            with st.spinner("Retrieving speeches and generating answer..."):
                result = generate_ecb_speech_response(
                    user_input=user_input,
                    conversation_history=conv_history,
                    audience_level=mapped_level,
                    documentation_needed=False,
                    mode="deep",
                    speaker_filter=selected_speaker,
                    year_filter=selected_year,
                    topic_filter=selected_topic,
                )

            if result and len(result) >= 6:
                response, speech_segments, rating, assessment, _, _ = result
            else:
                response, speech_segments, rating = result[0], [], 0

            # Tone extraction for display (recompute from full speeches if available)
            tone_line = ""
            if USE_SENTIMENT and speech_segments:
                try:
                    # Load sidecar to build full texts
                    with open(
                        "processed_ecb_data/speech_sidecar_token_chunks.json",
                        "r",
                        encoding="utf-8",
                    ) as f:
                        sidecar = json.load(f)
                    texts = []
                    for seg in speech_segments:
                        sid = seg.get("speech_id")
                        if sid and sid in sidecar:
                            full_txt = " ".join(c.get("text", "") for c in sidecar[sid])
                            texts.append(full_txt)
                    if texts:
                        tone_line = sentiment_facets_summary(texts)
                except Exception:
                    tone_line = ""

            if response:
                conv_history.append(
                    {
                        "role": "assistant",
                        "content": response,
                        "audience_level": mapped_level,
                        "speech_segments": speech_segments,
                        "rating": rating,
                        "tone_line": tone_line,
                    }
                )
                st.session_state.conversations[conv_id] = conv_history
                st.session_state.last_user_input = user_input
            else:
                st.error("No response generated. Please rephrase.")

    except Exception as e:
        st.error(f"Error: {e}")

elif submitted and not user_input:
    st.warning("Please enter a question before submitting.")

# Display conversation
conv_id = st.session_state.current_conversation_id
if conv_id in st.session_state.conversations:
    history = st.session_state.conversations[conv_id]
    for msg in history:
        if msg["role"] == "user":
            st.write(f"**You:** {msg['content']}")
        else:
            level = msg.get("audience_level", mapped_level)
            css_class = f"{level}-response"
            badge_class = (
                "general-badge" if level == "general" else "professional-badge"
            )
            badge_label = (
                "General audience" if level == "general" else "Professional audience"
            )
            html = f"""
            <div class="assistant-response {css_class}">
                <span class="audience-badge {badge_class}">{badge_label}</span><br><br>
                {msg['content']}
            </div>
            """
            st.markdown(html, unsafe_allow_html=True)

            # Sources expander
            if msg.get("speech_segments"):
                with st.expander("🔎 Source Information"):
                    for i, seg in enumerate(msg["speech_segments"][:6], 1):
                        if isinstance(seg, dict):
                            speaker = seg.get("speaker", "Unknown")
                            date = seg.get("date", "Unknown date")
                            title = seg.get("title", "")
                            st.write(
                                f"{i}. {speaker} — {date} {(' | ' + title) if title else ''}"
                            )
                        else:
                            st.write(f"{i}. {str(seg)[:100]}...")

            # Tone expander
            if msg.get("tone_line"):
                with st.expander("🎼 Tone Analysis"):
                    st.write(msg["tone_line"])

            if "rating" in msg and msg["rating"] > 0 and level == "professional":
                st.caption(f"Confidence score: {msg['rating']}/5")


