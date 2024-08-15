from langchain_community.vectorstores import Chroma
from langchain_community import embeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

model_local = ChatOllama(model="mistral")

persist_directory= 'db'

embedding=embeddings.OllamaEmbeddings(model='nomic-embed-text')

vectordb= Chroma(persist_directory=persist_directory,embedding_function=embedding)

retriever = vectordb.as_retriever()



print("\n#######\n After RAG\n")
after_rag_template ="""Answer the question based only on the following context:
{context}
Question: {question}
"""

after_rag_prompt = ChatPromptTemplate.from_template(after_rag_template)
after_rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | after_rag_prompt
    | model_local
    | StrOutputParser()
)

print(after_rag_chain.invoke("Write a python script that says hello world"))
