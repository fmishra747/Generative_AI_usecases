import os
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains import RetrievalQA
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

import faiss
import pickle

# 1. Download or scrape the OpenVINO documentation
loader = UnstructuredURLLoader(urls=["https://docs.openvino.ai/2024/index.html"])
data = loader.load()
print(data)

# 2. Index the documentation using LangChain and FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
rsplitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=200)
docs = rsplitter.split_documents(data)

# 3. Convert the document into embeddings
embeddings = OpenAIEmbeddings()
vectorindex_openapi = FAISS.from_documents(docs, embeddings)

# Save the FAISS vector index
vectorindex_openapi.save_local("faiss_openvino_index")


# Define the FastAPI app
app = FastAPI()

# Load the API key from environment variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# Model Input
class QueryModel(BaseModel):
    query: str  # Change this to 'query' to align with what RetrievalQA expects


# Initialize the OpenAI LLM
def get_openai_llm():
    if OPENAI_API_KEY is None:
        raise HTTPException(status_code=500, detail="OpenAI API key not found in environment variables.")
    return OpenAI(api_key=OPENAI_API_KEY)


# Initialize vector store (using FAISS)
def get_vectorstore():
    embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorestore = FAISS.load_local("faiss_openvino_index", embeddings, allow_dangerous_deserialization=True)
    return vectorestore

# Set up the RAG pipeline
def set_up_rag_pipeline():
    vectorstore = get_vectorstore()
    retriever = vectorstore.as_retriever()

    # Define prompt template
    prompt_template = """Use the following OpenVINO documentation to answer the question:
    {context}

    Question: {question}
    Answer: """
    
    # Create LLM Chain
    prompt = PromptTemplate(input_variables=["context", "question"], template=prompt_template)
    llm = get_openai_llm()
    llm_chain = LLMChain(prompt=prompt, llm=llm)

    # Create combined documents chain using StuffDocumentsChain
    combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context"
    )

    # Create RetrievalQA chain
    rag_chain = RetrievalQA(
        retriever=retriever,
        combine_documents_chain=combine_documents_chain
    )

    return rag_chain


# Endpoint to query RAG Chain
@app.post("/ask")
async def ask_question(query: QueryModel):
    try:
        # Set up the RAG pipeline
        rag_pipeline = set_up_rag_pipeline()

        # Get answer from RAG pipeline
        result = rag_pipeline({"query": query.query})  # Pass "query" as the key, since RetrievalQA expects it

        # Return the result
        return {"answer": result['result']}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Running the app using uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
