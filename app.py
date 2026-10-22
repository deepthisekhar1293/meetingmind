"""Streamlit web interface for MeetingMind. Run with `streamlit run app.py`."""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from streamlit.errors import StreamlitSecretNotFoundError

from extractor import (
    DEFAULT_MODEL,
    action_items_dataframe,
    extract_notes,
    to_csv,
    to_jira_csv,
    to_markdown,
)

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_PATH = PROJECT_DIR / "sample_data" / "meeting1.txt"

# Load the local secret before building the page; the key is never shown in the UI.
load_dotenv(PROJECT_DIR / ".env")
st.set_page_config(page_title="MeetingMind", layout="wide")
st.title("MeetingMind")
st.caption("Turn a meeting transcript into decisions, action items, risks, and open questions.")


def get_api_key() -> str:
    """Return the Claude key from local .env or hosted Streamlit secrets.

    The key is used by the server-side API client and is not displayed to users.
    """
    key = os.getenv("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    try:
        return str(st.secrets["ANTHROPIC_API_KEY"]).strip()
    except (KeyError, StreamlitSecretNotFoundError):
        return ""


with st.sidebar:
    st.header("Settings")
    model = st.text_input("Claude model", value=DEFAULT_MODEL)
    st.markdown("---")
    st.write("1. Paste or upload a transcript.")
    st.write("2. Extract structured meeting notes.")
    st.write("3. Review the results before exporting.")

# Streamlit gives us the uploaded file bytes; decode them and keep the transcript
# in this browser session so the user can review or edit it before sending it.
uploaded = st.file_uploader("Upload a UTF-8 .txt transcript", type=["txt"])
if uploaded is not None:
    upload_id = (uploaded.name, uploaded.size)
    if st.session_state.get("uploaded_id") != upload_id:
        try:
            st.session_state["transcript"] = uploaded.getvalue().decode("utf-8")
            st.session_state["uploaded_id"] = upload_id
        except UnicodeDecodeError:
            st.error("The uploaded file is not valid UTF-8. Save it as UTF-8 and try again.")

if st.button("Load sample meeting"):
    try:
        st.session_state["transcript"] = SAMPLE_PATH.read_text(encoding="utf-8")
        st.session_state.pop("uploaded_id", None)
    except OSError as error:
        st.error(f"Could not load the sample transcript: {error}")

# Pasting, uploading, and loading the sample all feed the same text area value.
transcript = st.text_area(
    "Meeting transcript",
    key="transcript",
    height=320,
    placeholder="Paste your meeting transcript here...",
)

if st.button("Extract notes", type="primary"):
    api_key = get_api_key()
    if not api_key:
        st.error(
            "No API key found. Set ANTHROPIC_API_KEY in your .env file "
            "or Streamlit secrets."
        )
    elif not model.strip():
        st.error("Enter a Claude model name in the sidebar.")
    else:
        st.session_state.pop("notes", None)
        try:
            import anthropic

            # The transcript is sent to Claude from the app server; the API key
            # stays on the server and is not sent to the browser.
            client = anthropic.Anthropic(api_key=api_key)
            with st.spinner("Analysing transcript..."):
                st.session_state["notes"] = extract_notes(
                    client,
                    transcript,
                    model=model.strip(),
                )
        except (ValueError, RuntimeError) as error:
            st.error(str(error))
        except anthropic.APIError as error:
            st.error(f"Claude API request failed: {error}")

notes = st.session_state.get("notes")
if notes is not None:
    # Display the validated response, then generate downloads from those notes.
    st.subheader("Summary")
    st.write(notes.summary)
    left, right = st.columns(2)
    with left:
        st.subheader("Decisions")
        for decision in notes.decisions or ["None"]:
            st.markdown(f"- {decision}")
        st.subheader("Open questions")
        for question in notes.open_questions or ["None"]:
            st.markdown(f"- {question}")
    with right:
        st.subheader("Risks / blockers")
        for risk in notes.risks_or_blockers or ["None"]:
            st.markdown(f"- {risk}")
    st.subheader("Action items")
    st.dataframe(action_items_dataframe(notes), hide_index=True)
    markdown_column, csv_column, jira_column = st.columns(3)
    markdown_column.download_button(
        "Download Markdown",
        to_markdown(notes),
        "meeting_notes.md",
        "text/markdown",
    )
    csv_column.download_button(
        "Download CSV",
        to_csv(notes),
        "action_items.csv",
        "text/csv",
    )
    jira_column.download_button(
        "Download Jira CSV",
        to_jira_csv(notes),
        "jira_import.csv",
        "text/csv",
    )
