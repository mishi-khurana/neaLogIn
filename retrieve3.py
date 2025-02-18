from langchain_core.messages import HumanMessage,SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from nltk.tokenize import word_tokenize
from flask import Flask, render_template, request, flash, redirect, url_for


app = Flask(__name__)
import mysql.connector
from spotipy.oauth2 import SpotifyClientCredentials
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from pprint import pprint
from time import sleep
from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from flask import Flask, json
from flask_mail import Mail, Message
import numpy as np
import re
from itsdangerous import URLSafeTimedSerializer
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
import nltk

nltk.download('punkt_tab')

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["song"] = record.get("song")
    metadata["song_id"] = record.get("song_id")
    return metadata

loader = JSONLoader(
    file_path='./data/songs.json',
    jq_schema='.[]',
    content_key="lyrics",
    metadata_func=metadata_func
    )

data = loader.load()

model = OllamaLLM(model="llama3.1", temperature=0.0)
embedding = OllamaEmbeddings(model="llama3.1")


# you are loading db using path here
new_db = FAISS.load_local("data/my_songs_db", embedding, allow_dangerous_deserialization=True)


results_with_scores = new_db.similarity_search("The song is about anger, frustration and desperation.", k=10)

for doc in results_with_scores:
    print(doc)

retriever = BM25Retriever.from_documents(results_with_scores, preprocess_func=word_tokenize)
result = retriever.invoke("anger, frustration and desperation")
print(result[0])


db = 'data/chroma_33words/'
vectordb = Chroma(persist_directory='data/chroma_33words/', embedding_function=embedding)   
vectordb = Chroma(
    persist_directory=db,
    embedding_function=embedding)

new_results = vectordb.similarity_search("The song is about anger, frustration and desperation.", k=10)

for r in new_results:
    print(r)
    
    