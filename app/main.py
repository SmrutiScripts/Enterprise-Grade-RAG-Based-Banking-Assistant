
import streamlit as st
from app.rag_pipeline import retrieve_context   

st.set_page_config(page_title="USBank RAG Chatbot", layout="centered")

st.title("USBank RAG Chatbot")


if "history" not in st.session_state:
    st.session_state.history = []


query = st.text_input("Ask your question:")

if st.button("Send") and query:
    response = retrieve_context(query)
  
    st.session_state.history.append({"user": query, "bot": response})


for chat in st.session_state.history:
    st.markdown(f"**You:** {chat['user']}")
    st.markdown(f"**Bot:** {chat['bot']}")
    st.markdown("---")
