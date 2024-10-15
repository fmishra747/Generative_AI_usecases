"""To build a Retrieval-Augmented Generation (RAG) pipeline using LangChain, which takes input from the OpenVINO documentation, stores it in a FAISS vector database, and retrieves relevant information for generating a tutorial based on the keyword "OpenVINO," we can follow the below steps.
Prerequisites:
FAStAPI :for building the API microservice
uvicorn: server for running the FATSAPI app
LangChain: A library for creating LLM-powered pipelines for retrieval and document processing 
FAISS: A fast similarity search tool for vectors, it is a vector store for document serach and similarity retrieval 
BeautifulSoup or Scrapy: For scraping the HTML content.
OpenAI/LLM: To generate the responses for language generation 
Python Libraries: Install required libraries."""
#1.SetUp the environment and add the needed libraries 
import os
os.environ['OPENAI_API_KEY']='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'
from langchain.llms import OpenAI

from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import  SimpleSequentialChain
from langchain.chains import  SequentialChain
from langchain.chains import RetrievalQA,RetrievalQAWithSourcesChain
from langchain.agents import AgentType, initialize_agent,load_tools
from langchain.memory import ConversationBufferMemory 
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from multiprocessing import Queue
from rouge_score import rouge_scorer

import faiss
import pickle

import dill


llm=OpenAI(temperature=0.9, max_tokens=500)

#2.Download or scrape the OpenVINO documentation 
loader=UnstructuredURLLoader(urls=["https://docs.openvino.ai/2024/index.html"])
#2.1 store the documenttaion in data and preprare it for indexing
data=loader.load()

len(data)


#3. Index the doucmenttaion using LangChain and FAISS

#3.1 Text Splitter to create chunks then perform merge closer to 4097 (depends on LLM) to be more efficient, to create overlapping chunks

from langchain.text_splitter import RecursiveCharacterTextSplitter
rsplitter=RecursiveCharacterTextSplitter(chunk_size=200,chunk_overlap=200)
docs=rsplitter.split_documents(data)

for i, doc in enumerate(docs):
    print(f"Document {i + 1}:")
    print(doc.page_content)
    
#3.2  Convert the document into embeddings.Create OpenAPI embeddings 

embeddings=OpenAIEmbeddings()
vectorindex_openapi=FAISS.from_documents(docs,embeddings)

# Storing vector index as a pickle file ->serialized file create in local 
vectorindex_openapi.save_local("faiss_store")

       
    
#Retriever
#chain=RetrievalQAWithSourcesChain.from_llm(llm=llm,retriever=vectorIndex.as_retriever())
query = "who is Farida Mishra"
query1 = "what is Accuracy Aware Qunatization"
chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=vectorindex_openapi.as_retriever(), input_key="query", return_source_documents=True)

#langchain.debug=True
result=chain(query)
#input_data = {"question": query}
#result = chain.invoke(input_data)
#result = chain(input_data)
print(result)

# Metric of RAG pipeline 
#ROUGE scores range from 0 to 1 (or 0 to 100 if scaled). A higher score means more overlap between the generated text and the reference.
scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
scores = scorer.score(query, query1)
print(f'ROUGE-1: {scores["rouge1"].fmeasure}')
print(f'ROUGE-L: {scores["rougeL"].fmeasure}')
# Make it a application RESTAPIs, docker 
          

