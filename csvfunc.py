import streamlit as st
from pathlib import Path
from streamlit_chat import message
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
import os
api_key = st.text_input('put your api key here',type='default')
os.environ["OPENAI_API_KEY"]=api_key

def save_csv_to_folder(uploaded_file, dest_dir):
    # Save uploaded file to destination folder.
    save_folder = dest_dir
    save_path = Path(save_folder, uploaded_file.name)
    with open(save_path, mode='wb') as w:
        w.write(uploaded_file.getvalue())
    if save_path.exists():
        st.success(f'File {uploaded_file.name} is successfully saved!')
        return save_path
    else:
        st.error(f"Failed to save file {uploaded_file.name}")
        return None


def generate_response(chain, user_query):
    response = chain({"question": user_query})
    return response['result']

"""
def run_chatbot():
    st.title('Langchain ChatBot for Your Custom CSV DATA POWERED BY OPEN AI')

    api_key = st.sidebar.text_input('Enter your Open AI API KEY', 'API KEY')
    destination_dir = st.sidebar.text_input('Enter your destination file dir', 'DIR')

    if st.sidebar.button("Set API KEY"):
        st.write("Your OPEN AI API KEY IS", api_key)
        os.environ["OPENAI_API_KEY"] = api_key

    csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")
    if csv_file_uploaded is not None:
        csv_file_path = save_csv_to_folder(csv_file_uploaded, destination_dir)
        if csv_file_path is not None:
            loader = CSVLoader(file_path=csv_file_path)
            index_creator = VectorstoreIndexCreator()
            docsearch = index_creator.from_loaders([loader])
            chain = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
            st.title("Chat with your CSV Data")
            if 'generated' not in st.session_state:
                st.session_state['generated'] = []
            if 'past' not in st.session_state:
                st.session_state['past'] = []
            user_input = st.text_input("You: ","Ask Question From your Document?", key="input")
            if user_input:
                output = generate_response(chain, user_input)
                st.session_state.past.append(user_input)
                st.session_state.generated.append(output)
            if st.session_state['generated']:
                for i in range(len(st.session_state['generated'])-1, -1, -1):
                    message(st.session_state["generated"][i], key=str(i))
                    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
"""