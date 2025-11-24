# execute only when you want to add data to the pinecone 

from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings=download_hugging_face_embeddings()


if PINECONE_API_KEY is None:
    raise ValueError("PINECONE_API_KEY is not set. Please set the environment variable before running this cell.")

pc = Pinecone()
index_name = "medibot"

pc.create_index(
    name=index_name,
    dimension=384,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

docsearch = Pinecone.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)