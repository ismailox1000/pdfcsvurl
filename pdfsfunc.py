import atexit
import shutil
import streamlit as st
import os
#from langchain import OpenAI
from streamlit_chat import message 
from utils import get_initial_message, update_chat
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
#from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
#from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import DirectoryLoader
#import openai

# Register cleanup function
def cleanup_tempdir(directory_path):
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)

def create_chat_model():
    # Set OpenAI API key
    
    # Create chat model
    llm = ChatOpenAI(Temperature=0.7, model_name='gpt-3.5-turbo')
    return llm
#uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx"], accept_multiple_files=True)

def upload_files(directory_path,uploaded_files):
# File upload
#


    # Create a temporary directory to store the uploaded files
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Created directory: {directory_path}")
    else:
        print(f"Directory already exists: {directory_path}")
    for file in uploaded_files:
        with open(os.path.join('tempDir',file.name), "wb") as f:
            f.write(file.getbuffer())
    
    pdf_loader = DirectoryLoader(directory_path, glob="**/*.pdf")
    word_loader = DirectoryLoader(directory_path, glob="**/*.docx")
    st.success('File Saved')
    
    # Load documents
    loaders=[pdf_loader,word_loader]
    documents=[]
    for loader in loaders: 
        documents.extend(loader.load())
    print(f'total number of docs is:{len(documents)}')
    
    # Split documents
    text_splitter=CharacterTextSplitter(chunk_size=1000,chunk_overlap=200,length_function=len)    
    documents=text_splitter.split_documents(documents)
    
    # Create vector store and embeddings
    embeddings=OpenAIEmbeddings()
    vectorstore=Chroma.from_documents(documents,embeddings)
    
    # Create memory and conversational retrieval chain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa=ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0),vectorstore.as_retriever(),memory=memory) 

    return qa, vectorstore

def run_chat():
    # Define directory for uploaded files
    directory_path = "./tempDir"

    # Register cleanup function
    atexit.register(cleanup_tempdir, directory_path)

    llm = create_chat_model()
    
    qa, vectorstore = upload_files(directory_path)

    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []

    query = st.text_input("Type Your Question:",key="input")

    if "messages" not in st.session_state:
        st.session_state["messages"] = get_initial_message()

    if query:
        with st.spinner("generating..."):
            messages = st.session_state["messages"]
            messages = update_chat(messages, "user", query)
            result=qa({"question": query})
            response = result['answer']
            messages = update_chat(messages, "assistant", response)
            st.session_state.past.append(query)
            st.session_state.generated.append(response)

    if st.session_state["generated"]:
        for i in range(len(st.session_state["generated"]) - 1, -1, -1):
            message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
            message(st.session_state["generated"][i], key=str(i))


        with st.expander("Show Messages"):
            st.write(messages)

    if not st.session_state:
        if os.path.exists(directory_path):
            for file in os.listdir(directory_path):
                os.remove(os.path.join(directory_path, file))
            os.rmdir(directory_path)
