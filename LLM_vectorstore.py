from langchain_community.vectorstores import Chroma
from langchain_community import embeddings

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import DirectoryLoader




loader = DirectoryLoader('./docs/', glob="./*.pdf", loader_cls=PyPDFLoader)

documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

texts = text_splitter.split_documents(documents)

embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')
persist_directory = 'db'


vectordb = Chroma.from_documents(documents=texts,embedding=embedding,persist_directory=persist_directory)

vectordb.persist()
vectordb = None

vectordb = Chroma(persist_directory=persist_directory,
                  embedding_function=embedding)

