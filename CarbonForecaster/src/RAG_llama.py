import os
import openai
import sys
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Environment variables
sys.path.append('../..')
_ = load_dotenv(find_dotenv()) # read local .env file
openai.api_key = os.environ['OPENAI_API_KEY']

# Document loading
loader = PyPDFLoader("sources/RioTinto-AR-2023.pdf")
pages = loader.load()

# Document splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
splits = text_splitter.split_documents(pages)

# Vector stores and embeddings
embedding = OpenAIEmbeddings(api_key = openai.api_key)
persist_directory = 'vector_db/chroma/'

vectordb = Chroma.from_documents(
    documents=splits,
    embedding=embedding,
    persist_directory=persist_directory
)

vectordb.persist()
