from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings

MODEL = "tinyllama"

model = Ollama(model=MODEL)
embeddings = OllamaEmbeddings(model=MODEL)
model.invoke("what is machine learning in a few words?")