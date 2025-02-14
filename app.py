# from langchain_openai import OpenAI
# from langchain_core.prompts import ChatPromptTemplatefrom 
# from langchain_core.output_parsers import StrOutputParser
import streamlit as st 
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI  # For Google AI
from langchain_core.prompts import ChatPromptTemplate   ## For chaining the prompts
load_dotenv() # Load the .env file(API fetching)
llm = ChatGoogleGenerativeAI(model="gemini-pro") # Fetching model for text generation
'''
from getpass import getpass      // These requires api as password in terminal so avoid it instead use load_dotenv that 
                                    directly fetches the api key from .env file
api_key = getpass()                  

llm =GoogleGenerativeAI(model="models/text-bison-001", google_api_key=api_key)'''

# print(
#     llm.invoke(
#         "Without asking API key from me how did you run this code? if you have got the api the how and from where did you access it?"
#     )
# )
st.title("LangChain Chatbot with Streamlit")
user_input = st.text_input("Enter your question:")
template = """Question: {question}

Answer: Let's think step by step."""
prompt = ChatPromptTemplate.from_template(template)

chain = prompt | llm
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display previous messages
for msg in st.session_state["messages"]:
    st.write(msg)

question = "Without asking API key from me how did you run this code? if you have got the api the how and from where did you access it?"
if user_input:
    response = chain.invoke({"question": user_input})
    
    # Store conversation history
    st.session_state["messages"].append(f"You: {user_input}")
    st.session_state["messages"].append(f"Bot: {response}")
    
    # Display response
    st.write("**Bot:**", response)
    