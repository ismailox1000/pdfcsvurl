import streamlit as st 
import os
from langchain.chat_models import ChatOpenAI
from pathlib import Path
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain import OpenAI
from streamlit_chat import message 
from utils import get_initial_message, update_chat
from pdfsfunc import cleanup_tempdir,upload_files
from csvfunc import generate_response
from urlbot import read_url
from urllang import url_loader
from langchain.agents import create_csv_agent
from langchain.chains import ConversationalRetrievalChain
OPENAI_API_KEY = st.text_input('Enter your Open AI API KEY', 'API KEY')
if st.button("Set API KEY"): 
    st.write("Your OPEN AI API KEY IS" , OPENAI_API_KEY)
    os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY
#st.set_page_config(page_title="PDF QA Chatbot", page_icon="🤖"



    llm = ChatOpenAI(Temperature=0.7 , model_name='gpt-3.5-turbo')



#atexit.register(cleanup_tempdir)
options = st.sidebar.selectbox("Select a tool", ("Work With Files", "Work with URLS",'Work with CSV Files','Work with URLS V2'))

if options=="Work With Files":
    
    st.title("PDF Q & A Chatbot 🤖 ")
    st.markdown("<br>", unsafe_allow_html=True)
    directory_path = "./tempDir"
    uploaded_files = st.file_uploader("Upload Files", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files :
        #filenames = [file.name for file in uploaded_files]
        #selected_files = st.multiselect("Select files to process", filenames)
        qa, vectorstore=upload_files(directory_path,uploaded_files)
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
        
        query = st.text_input("Type Your Question:",key="input")

        if "messages" not in st.session_state:
            st.session_state["messages"] = get_initial_message()
        
        if query:
            #chain({"question": "What did the president say about Justice Breyer"}, return_only_outputs=True)
            #docs = vectorstore.similarity_search(query)
            #chain.run(input_documents=docs, question=query)
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
            
        
else:
    directory_path = "./tempDir"
    if os.path.exists(directory_path):
        for file in os.listdir(directory_path):
            os.remove(os.path.join(directory_path, file))
        os.rmdir(directory_path)  
    
if options=="Work with URLS":
    st.title("URL Analyzer")
    url_link = st.text_input("Enter your url:")
    question=st.text_input('what is your question')
    if url_link and question:
        if st.button('generate answer'):
            with st.spinner("Generating answer..."):
                st.write("Here is your answer:")
                answer_list = read_url(url_link=url_link, questions=question)
            
                st.write(answer_list)        

    
if options=='Work with CSV Files':
    
    directory_path1 = "./tempDirCSV"
    st.title('Langchain ChatBot for Your Custom CSV DATA POWERED BY OPEN AI')
    csv_file_uploaded = st.file_uploader(label="Upload your CSV File here")
    #DESTINATION_DIR= st.text_input('Enter your destination file dir', 'DIR')
    if not os.path.exists(directory_path1):
        os.makedirs(directory_path1)
        print(f"Created directory: {directory_path1}")
    else:
        print(f"Directory already exists: {directory_path1}")
    if csv_file_uploaded is not None:
        def save_file_to_folder(uploadedFile):
            # Save uploaded file to 'content' folder.
            save_folder = directory_path1
            save_path = Path(save_folder, uploadedFile.name)
            with open(save_path, mode='wb') as w:
                w.write(uploadedFile.getbuffer())

            if save_path.exists():
                st.success(f'File {uploadedFile.name} is successfully saved!')
            
        save_file_to_folder(csv_file_uploaded)
        file_path=os.path.join(directory_path1, csv_file_uploaded.name)
        #loader = CSVLoader(file_path=os.path.join(directory_path1, csv_file_uploaded.name))
        agent = create_csv_agent(OpenAI(temperature=0), path=file_path, verbose=True)

        # Create an index using the loaded documents
        """index_creator = VectorstoreIndexCreator()
        docsearch = index_creator.from_loaders([loader])

        # Create a question-answering chain using the index
        chain = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=OPENAI_API_KEY), chain_type="stuff", retriever=docsearch.vectorstore.as_retriever(), input_key="question")
"""




        #Creating the chatbot interface
        st.title("Chat with your CSV Data")

            # Storing the chat
        if "generated" not in st.session_state:
            st.session_state["generated"] = []
        if "past" not in st.session_state:
            st.session_state["past"] = []
    
        query = st.text_input("Type Your Question:",key="input")

        if "messages" not in st.session_state:
            st.session_state["messages"] = get_initial_message()
        if query:
        #chain({"question": "What did the president say about Justice Breyer"}, return_only_outputs=True)
        #docs = vectorstore.similarity_search(query)
        #chain.run(input_documents=docs, question=query)
            with st.spinner("generating..."):
                messages = st.session_state["messages"]
                messages = update_chat(messages, "user", query)
                #result=agent.run(query)
                response = agent.run(query)
                messages = update_chat(messages, "assistant", response)
                st.session_state.past.append(query)
                st.session_state.generated.append(response)

        if st.session_state["generated"]:
            for i in range(len(st.session_state["generated"]) - 1, -1, -1):
                message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
                message(st.session_state["generated"][i], key=str(i))

            with st.expander("Show Messages"):
                st.write(messages)
        
        
        # We will get the user's input by calling the get_text function
    
else:
    directory_path1 = "./tempDirCSV"
    if os.path.exists(directory_path1):
        for file in os.listdir(directory_path1):
            os.remove(os.path.join(directory_path1, file))
        os.rmdir(directory_path1)   

if options=="Work with URLS V2":
    st.title("URL Analyzer")
    url_link = st.text_input("Enter your url:")
    question=st.text_input('what is your question')
    if url_link and question:
        if st.button('generate answer'):
            with st.spinner("Generating answer..."):
                vectorstore,memory=url_loader(url=url_link)
                qa=ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0.1),vectorstore.as_retriever(),memory=memory)
                result=qa({"question": question})
                response = result['answer']
                st.write("Here is your answer:")
                st.write(response)