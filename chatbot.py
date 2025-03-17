import os
import sys

# Force CPU mode - add these before other imports
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["USE_CPU"] = "1" 
os.environ["FORCE_CPU"] = "1"
os.environ["TORCH_DEVICE"] = "cpu"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Check if .no_cuda exists (created by fix_cuda.py)
if os.path.exists(".no_cuda"):
    print("Running in CPU-only mode")

# Now import the rest of your libraries
import streamlit as st
from datetime import datetime
import torch
if hasattr(torch, 'cuda'):
    torch.cuda.is_available = lambda: False

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint

DB_FAISS_PATH = "vectorstore/db_faiss"

@st.cache_resource
def get_vectorstore():
    """Loads the FAISS vector store with sentence embeddings."""
    try:
        embedding_model = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            encode_kwargs={'device': 'cpu', 'use_gpu': False}
        )
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"🚨 Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    """Creates a custom prompt template."""
    return PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

def load_llm(huggingface_repo_id, HF_TOKEN):
    """Loads the Hugging Face LLM for inference."""
    try:
        return HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={
                "token": HF_TOKEN,
                "max_length": "512",
                "device": "cpu",
                "use_gpu": False
            }
        )
    except Exception as e:
        st.error(f"🚨 Error loading LLM: {str(e)}")
        return None

def reset_chat():
    """Saves the current chat to history with timestamp and starts a new one."""
    if "messages" in st.session_state and st.session_state.messages:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.chat_history.append({"timestamp": timestamp, "messages": st.session_state.messages})
    st.session_state.messages = []  # Start a fresh conversation

def restore_chat(index):
    """Restores a selected chat from history."""
    if 0 <= index < len(st.session_state.chat_history):
        st.session_state.messages = st.session_state.chat_history[index]["messages"]

def main():
    """Main function to run the Streamlit chatbot."""
    st.set_page_config(page_title="🦷 Dental Assistant AI", page_icon="🦷", layout="wide")

    # Initialize session states
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # 🏷️ Title and "New Chat" button
    st.title("🦷 Dental Assistant AI")
    col1, col2 = st.columns([3, 1])
    with col2:
        st.button("🆕 New Chat", on_click=reset_chat)  # Button to start a new chat session

    # 📜 Sidebar for Chat History
    with st.sidebar:
        st.title("📜 Chat History")
        if len(st.session_state.chat_history) > 0:
            for idx, chat in enumerate(st.session_state.chat_history):
                with st.expander(f"Chat {idx + 1} ({chat['timestamp']})"):
                    preview = " | ".join([msg['content'][:50] for msg in chat['messages'][:3]]) + "..."
                    st.text(preview)  # Show preview of first few messages
                    if st.button(f"🔄 Restore Chat {idx + 1}", key=f"restore_{idx}"):
                        restore_chat(idx)
                        st.rerun()
        else:
            st.info("No chat history yet. Start a conversation!")

    # 💬 Display chat messages
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])

    prompt = st.chat_input("Type your question here...")

    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})

        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer the user's question.
            If you don’t know the answer, just say that you don’t know, don’t try to make up an answer.
            Don’t provide anything out of the given context.

            Context: {context}
            Question: {question}

            Start the answer directly. No small talk, please.
        """
        
        HUGGINGFACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"
        HF_TOKEN = os.environ.get("HF_TOKEN")

        try:
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load the vector store")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=load_llm(huggingface_repo_id=HUGGINGFACE_REPO_ID, HF_TOKEN=HF_TOKEN),
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={'k': 3}),
                return_source_documents=True,
                chain_type_kwargs={'prompt': set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)}
            )

            response = qa_chain.invoke({'query': prompt})

            # 📌 Format response and sources
            result = response["result"].replace('\n', ' ').strip()
            source_documents = response["source_documents"]
            source_docs_text = "\n\n**📚 Source Documents:**\n"
            for i, doc in enumerate(source_documents, 1):
                content = doc.page_content.replace('\n', ' ').strip()
                source_docs_text += f"{i}. {content}\n"

            result_to_show = f"{result}\n{source_docs_text}"
            
            st.chat_message('assistant').markdown(result_to_show)
            st.session_state.messages.append({'role': 'assistant', 'content': result_to_show})

        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")

if __name__ == "__main__":
    main()
