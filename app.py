from flask import Flask, request, session

import datetime

from BookExplorer import *
import pickle

app = Flask(__name__)
app.secret_key = "*ajagb1294$25#!2"

app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(minutes=10)


@app.route("/")
def index():
    return "Welcome to Book Explorer API"


@app.route("/predict", methods=["POST"])
def predict():
    """
    Make prediction from a query
    """
    data = request.json 
    if "model" not in session: # if model does not exist, create one
        if "file_path" in data:
            file_path = data["file_path"]
            session["model"] = 1
            session.permanent = True
            model = BookExplorer(file_path=file_path)
            test_docs = model.test_documents
        else:
            web_path = data["web_path"]
            session["model"] = 1
            session.permanent = True
            model = BookExplorer(web_path=web_path)
            test_docs = model.test_documents
    
    vectorstore = Chroma(persist_directory="./chroma_db", 
                         embedding_function=HuggingFaceEmbeddings())

    if "query" in data:
        query = data["query"]
        retrieved_docs = get_relevant_docs(vectorstore, query)
        resp = generate_response(document_chain, query, retrieved_docs)
        test_docs = None
    else:
        resp = "Initialized"
    

    return {"response": resp, "test": test_docs}