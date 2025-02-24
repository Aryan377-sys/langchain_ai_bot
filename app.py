import streamlit as st
import os
import requests  # Fetching data from API
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings  # Updated Embeddings
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document

# Load environment variables (API Keys)
load_dotenv()

# Initialize LLM (Using Google Gemini)
llm = ChatGoogleGenerativeAI(model="gemini-pro")

# Streamlit UI
st.title("💬 CHATBOT FOR YOUR ALL QUERIES")

# API URL for fetching `.txt` file
API_URL = "https://datasets-server.huggingface.co/rows?dataset=bitext%2FBitext-customer-support-llm-chatbot-training-dataset&config=default&split=train&offset=0&length=100"

# Fetch the .txt data from API
response = requests.get(API_URL)
if response.status_code == 200:
    customer_care_text = response.text
    st.success("📂 Customer care data loaded successfully from API!")
else:
    st.error("⚠️ Failed to fetch the customer care text from API!")
    st.stop()

# Convert fetched text into LangChain Documents
documents = [Document(page_content=customer_care_text)]

# Split text into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# ✅ Using Hugging Face Embeddings instead of OpenAI
embedding_function = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create FAISS VectorStore
vectorstore = FAISS.from_documents(chunks, embedding_function)

# Create retriever
retriever = vectorstore.as_retriever()

# Structured Prompt
structured_prompt = ChatPromptTemplate.from_template("""
### Customer Support Chatbot ###

You are a **customer service AI assistant**. Given the following **retrieved context**, answer the user's question **step-by-step**.

**Context:** {context}
**User Question:** {question}

Your answer should be **polite, well-structured, and helpful**.
""")

# Create RAG Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm, retriever=retriever, chain_type="stuff", return_source_documents=True
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    st.write(msg)

# User input
user_input = st.text_input("Ask a customer support question:")
if user_input:
    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(user_input)
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs]) if retrieved_docs else "No relevant information found."

    # ✅ Define prompt correctly before using it
    prompt = f"""
    You are a customer service chatbot. Answer the following question based on the given information:

    Context: {context_text}
    User Question: {user_input}

    Provide a concise, polite, and well-structured response.
    """

    # ✅ Query Google Gemini and extract text properly
    response_obj = llm.invoke(prompt)
    response_text = response_obj.content  # Extract actual text response

    # Store conversation history
    st.session_state["messages"].append(f"**You:** {user_input}")
    st.session_state["messages"].append(f"**Bot:** {response_text}")

    # Display response
    st.write("**Bot:**", response_text)
