from flask import Flask,render_template,jsonify,request
from threading import Thread
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import Pinecone
from langchain_openai import OpenAI
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app=Flask(__name__)

load_dotenv()

# Read API keys from environment and only set them if present (avoid setting None)
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

if PINECONE_API_KEY:
    os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Lazy initialization to avoid long-running or failing work at import time
embeddings = None
docsearch = None
retirver = None
llm = None
prompt = None
question_answer_chain = None
rag_chain = None

def init_rag():
    """Initialize embeddings, Pinecone index, retriever and chains on first use."""
    global embeddings, docsearch, retirver, llm, prompt, question_answer_chain, rag_chain
    if rag_chain is not None:
        return

    embeddings = download_hugging_face_embeddings()

    index_name = "medibot"
    docsearch = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )

    retirver = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = OpenAI(temperature=0.4, max_tokens=500)
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retirver, question_answer_chain)


# Start initialization in background so the web server can bind quickly
def _bg_init():
    try:
        init_rag()
        app.logger.info("RAG initialization completed")
    except Exception:
        app.logger.exception("RAG initialization failed")

Thread(target=_bg_init, daemon=True).start()


@app.route("/")
def index():
    return render_template("chat.html")


# @app.route("/get",methods=["GET","POST"])
# def chat():
#     msg=request.form["user_input"]
#     input=msg
#     print(input)
#     response=rag_chain.invoke({"input":msg})
#     print("Response:",response["answer"])
#     return str(response["answer"])

@app.route("/get",methods=["GET","POST"])
def chat():
    # CHANGE 1: Use .get() to safely retrieve the form data. 
    # This returns None if the field is missing, preventing the KeyError.
    msg = request.form.get("user_input")
    
    # CHANGE 2: Add a check to return a graceful response if the message is empty.
    if not msg:
        # A 400 status indicates a client-side error (bad request)
        return jsonify({"response": "Error: Please enter a message."}), 400 

    # Ensure RAG initialization (fast no-op if already initialized)
    try:
        init_rag()
    except Exception:
        app.logger.exception("init_rag() failed during request")

    if rag_chain is None:
        return jsonify({"response": "Service initializing, try again shortly."}), 503

    input = msg
    app.logger.info("Received user input")
    try:
        response = rag_chain.invoke({"input": msg})
    except Exception:
        app.logger.exception("RAG invocation failed")
        return jsonify({"response": "Internal server error"}), 500

    app.logger.info("RAG responded")
    return jsonify({"response": response["answer"]}), 200



if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080,debug=True)
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=True)


@app.route("/health")
def health():
    if rag_chain is None:
        return ("", 503)
    return ("", 200)