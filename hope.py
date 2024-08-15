import streamlit as st
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.text_splitter import CharacterTextSplitter
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
from hashlib import sha256
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY")
st.header("Chat with the Streamlit docs 💬 📚")

if "messages" not in st.session_state.keys():  # Initialize the chat message history
    st.session_state.messages = [
        {"role": "assistant", "content": "I will help you with the provided docs. Ask me anything"}
    ]

# Define the path to the local storage for ChromaDB
LOCAL_CHROMA_DB_PATH = "./local_chroma_db"

# Ensure the directory exists
os.makedirs(LOCAL_CHROMA_DB_PATH, exist_ok=True)

def hash_it(content: str):
    hash_obj = sha256()
    hash_obj.update(content.encode('utf-8'))
    return hash_obj.hexdigest()

def load_data():
    with st.spinner(text="Loading and indexing the Streamlit docs – hang tight! This should take 1-2 minutes."):

        # Incase the database does not exist, create it
        if not os.listdir(LOCAL_CHROMA_DB_PATH):

            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma(embedding_function=embeddings, persist_directory=LOCAL_CHROMA_DB_PATH)

            # Load documents
            loader = DirectoryLoader(path="./data")

            text_splitter = CharacterTextSplitter(
                separator="\n\n",
                chunk_overlap=100,
                length_function=len,
                chunk_size=800,
                is_separator_regex=False,
            )

            pages = loader.load_and_split(text_splitter)
            sources = [page.metadata['source'] for page in pages]
            _hash = [hash_it(source) for source in sources]
            vector_store.add_documents(pages, id=_hash)
        else: # Use the existing database
            embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            vector_store = Chroma(embedding_function=embeddings, persist_directory=LOCAL_CHROMA_DB_PATH)

        return vector_store

# Load the data
vector_store = load_data()

# Define the chat engine
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
chat_engine = RetrievalQA.from_llm(llm, retriever=vector_store.as_retriever(search_kwargs={"k": 5}))

# Define the chat template
template = """
You are a helpful AI assistant.
Answer based on the context provided. 
context: {context}. Everything in context is about IIITB.
input: {input}
answer:
"""
prompt = PromptTemplate.from_template(template)
combine_docs_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(vector_store.as_retriever(), combine_docs_chain)

# Load a sentence transformer model for similarity checking
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

def calculate_similarity(prompt, response):
    prompt_embedding = similarity_model.encode([prompt])
    response_embedding = similarity_model.encode([response])
    return cosine_similarity(prompt_embedding, response_embedding)[0][0]

def is_response_relevant(prompt, response, threshold=0.3):
    # Calculate similarity score
    similarity_score = calculate_similarity(prompt, response)
    return similarity_score >= threshold

if prompt := st.chat_input("Your question"):  # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:  # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # First try retrieval-based QA
            response = retrieval_chain.invoke({"input": f"{prompt}"})
            answer = response['answer']

            # Check if the response is relevant, if not, fall back to general LLM knowledge
            if not is_response_relevant(prompt, answer):
                answer = llm(f"{prompt}")['text']
            
            st.write(answer)
            message = {"role": "assistant", "content": answer}
            st.session_state.messages.append(message)  # Add response to message history
