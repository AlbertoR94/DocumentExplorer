from langchain_community.document_loaders import PDFMinerLoader
from langchain_community.document_loaders import WebBaseLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain_community.llms import HuggingFaceHub
# from langchain import PromptTemplate

with open("API_TOKEN.txt", "rb") as file:
    api_token = file.read()


import os

# Define prompt template here.
prompt = ChatPromptTemplate.from_template("""Answer the following question based on the provided context.

<context>
{context}
</context>

Question: {input}""")

class BookExplorer():
    def __init__(self, file_path=None, web_path=None, 
                 parent_chunk_size=735):
        self.FILE_PATH = file_path
        self.WEB_PATH = web_path
        self.PARENT_CHUNK_SIZE = parent_chunk_size
        #self.HUGGINGFACEHUB_API_TOKEN = huggingfacehub_api_token
        #self.CHILD_CHUNK_SIZE = child_chunk_size
        #self.CHILD_OVERLAP = child_overlap

        # This text splitter creates larger chunks of context
        self.parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.PARENT_CHUNK_SIZE)

        self.load_data()
        self.split_documents()
        self.create_vector_store()
        # self.create_model_instance()


    def load_data(self):
        """
        Load document from a file path or web path provided at initialization
        """
        
        if self.FILE_PATH:
            loader = PDFMinerLoader(self.FILE_PATH)
            self.pages = loader.load()
        if self.WEB_PATH:
            loader = WebBaseLoader(self.WEB_PATH)
            self.pages = loader.load()
    
    
    def split_documents(self):
        """
        Split documents using a text splitter instance.
        """
        self.documents = self.parent_splitter.split_documents(self.pages)
        num_docs = int(2500 / self.PARENT_CHUNK_SIZE)
        self.test_documents = [doc.page_content for doc in self.documents[0:num_docs]]

    
    def create_vector_store(self):
        """
        Create a vector store for embeddings
        """
        self.vectorstore = Chroma.from_documents(self.documents, 
                                                 HuggingFaceEmbeddings(), persist_directory="./chroma_db")



# create instance of LLM
repo_id = "declare-lab/flan-alpaca-large"
llm = HuggingFaceHub(repo_id=repo_id, 
                     model_kwargs={"temperature": 0.1, 
                                  "max_length": 200}, 
                     huggingfacehub_api_token=api_token)

document_chain = create_stuff_documents_chain(llm, prompt)

    
def get_relevant_docs(vectorstore, query, num_responses=10):
    """
    Generate most relevant documents given a query.
    """
    retrieved_docs = vectorstore.similarity_search(query, k=num_responses)
    return retrieved_docs
    
    
def generate_response(document_chain, query, retrieved_docs):
    """
    Generate a response from a query and relevant documents.
    """
    resp = document_chain.invoke({
        "input": query,
        "context": retrieved_docs
    })
    return resp
