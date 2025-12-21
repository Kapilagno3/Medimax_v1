from flask import Flask,render_template,jsonify,request
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

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY
os.environ["OPENAI_API_KEY"]=OPENAI_API_KEY

embeddings=download_hugging_face_embeddings()


index_name = "medibot"

# embed each chunk
docsearch=Pinecone.from_existing_index(
    index_name=index_name,
    embedding=embeddings,
)

retirver=docsearch.as_retriever(search_type="similarity",search_kwargs={"k":3})


llm=OpenAI(temperature=0.4,max_tokens=500)
prompt=ChatPromptTemplate.from_messages(
    [
        ("system",system_prompt),
        ("human","{input}")
    ]
)

question_answer_chain=create_stuff_documents_chain(llm,prompt)
rag_chain=create_retrieval_chain(retirver,question_answer_chain)


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

    input=msg
    print(input)
    response=rag_chain.invoke({"input":msg})
    print("Response:",response["answer"])
    
    # CHANGE 3: The frontend needs JSON data for easy updates, not just a string.
    # Return a JSON object with the response and status code 200 (OK).
    return jsonify({"response": response["answer"]}), 200



if __name__=="__main__":
    # app.run(host="0.0.0.0",port=8080,debug=True)
    app.run(host="0.0.0.0",debug=True)