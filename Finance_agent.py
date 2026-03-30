from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def create_knowledge_base():
    loader = TextLoader("Financial_data.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    knowledge_base = Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory="./vector_store"
    )
    print(f"Knowledge base created with {len(chunks)} chunks")
    return knowledge_base

def create_agent(knowledge_base):
    template = """You are a financial assistant specialized in ERP systems and Dynamics 365.
Analyze the available financial data and respond in a clear, executive manner.
If you detect risks or alerts, mention them explicitly.
If you don't have enough information, state it honestly.

Financial context:
{context}

Question: {question}

Answer (clear and business-decision oriented):"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    retriever = knowledge_base.as_retriever(search_kwargs={"k": 3})

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain

if __name__ == "__main__":
    knowledge_base = create_knowledge_base()
    agent = create_agent(knowledge_base)

    question = "Which invoices are overdue and what is the risk for cash flow?"
    response = agent.invoke(question)
    print("\nAgent response:")
    print(response)