import streamlit as st
from Finance_agent import create_knowledge_base, create_agent

st.set_page_config(
    page_title="ERP Financial AI Assistant",
    page_icon="💼",
    layout="wide"
)

st.title("ERP Financial AI Assistant")
st.caption("Powered by RAG + GPT")

if "agent" not in st.session_state:
    with st.spinner("Loading financial data..."):
        kb = create_knowledge_base()
        st.session_state.agent = create_agent(kb)
    st.success("Agent ready")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

question = st.chat_input("Ask about your financial data...")

if question:
    with st.chat_message("user"):
        st.write(question)
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            response = st.session_state.agent.invoke(question)
            st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.header("Sample questions")
    examples = [
        "Which invoices are overdue?",
        "Which department is over budget?",
        "What is our current liquidity?",
        "Are there any urgent financial alerts?",
        "Summarize the Q4 financial status"
    ]
    for e in examples:
        st.code(e)