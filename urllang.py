from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
import os
from langchain.memory import ConversationBufferMemory

#os.environ['OPENAI_API_KEY'] = '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'
def url_loader(url):
    loader = WebBaseLoader(url)
    data = loader.load()
    print(data)
    embeddings = OpenAIEmbeddings()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0,separator="\n")
    ruff_texts = text_splitter.split_documents(data)
    vectorstore = Chroma.from_documents(ruff_texts, embeddings)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    return vectorstore,memory