# ERP Financial AI Assistant

Conversational AI agent that answers questions about a financial ERP data using RAG (Retrieval-Augmented Generation).

## What it does
- Detects overdue invoices and cash flow risks
- Identifies departments over budget
- Summarizes financial status in natural language
- Answers questions about accounts receivable, budgets, and cash flow

## Architecture
- **RAG** — connects financial documents to an LLM for context-aware responses
- **LangChain** — orchestrates the retrieval and generation pipeline
- **ChromaDB** — vector database for semantic search
- **OpenAI GPT-3.5** — language model for response generation
- **Streamlit** — web interface

## How to run
1. Clone the repo
2. Create a virtual environment and install dependencies
```
pip install -r requirements.txt
```
3. Add your OpenAI API key to a `.env` file
```
OPENAI_API_KEY=your-key-here
```
4. Run the app
```
streamlit run app.py
```

## Next steps
- Connect to real D365 data via Dataverse API
- Add Azure AD authentication
- Implement response quality evaluation with RAGAS

## Stack
Python · LangChain · ChromaDB · OpenAI API · Streamlit