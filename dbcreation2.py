from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
import numpy as np
import json

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["song"] = record.get("song")
    metadata["song_id"] = record.get("song_id")
    return metadata


model = OllamaLLM(model="llama3.1", temperature=0.0)
persist_directory = 'data/chroma_33words/'
embedding = OllamaEmbeddings(model="llama3.1")



loader = JSONLoader(
    file_path='./data/songs.json',
    jq_schema='.[]',
    content_key="lyrics",
    metadata_func=metadata_func

    )

data = loader.load()


def generateData(data):
    
    for d in data:
        summary = model.invoke([SystemMessage(content="Summarise the lyrics of this song in a list of 5 words in this format: Sad, Regret, Betrayal. The output should only contain the five words")]+ #content of what it knows
                                      [HumanMessage(content=d.page_content)]).strip() #asking it based on its knowledge
        
        genre = model.invoke([SystemMessage(content="Extract a list of up to 5 genres of this song in this format: Pop, Rock, Folk. The output should only contain the five words")]+
                                      [HumanMessage(content=d.page_content)]).strip()
        
        song_id = d.metadata.get("song_id")
    
        d.metadata.update({
        "summary": summary,
        "genre": genre,
        "song_id": song_id
        })

        print(f"data is + {d.metadata}")


def store_embeddings(data):
    # Create structured text for embedding
    for d in data:
        embedding_text = f"Genre: {d.metadata['genre']}. Keywords: {d.metadata['summary']}. Song ID: {d.metadata['song_id']}"
        d.page_content = embedding_text  # Replace lyrics with structured text

    # Store in vector DB
    vectordb = Chroma.from_documents(documents=data, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()  
    return vectordb, embedding


generateData(data)  # Call the function to process metadata

vectordb = store_embeddings(data)


documents = vectordb._collection.get_all()

# Print the first document's metadata and embedding
if documents:
    first_doc = documents[0]  # Access the first document
    print("First Document Metadata:", first_doc.metadata)
    print("First Document Embedding:", first_doc.embedding)
else:
    print("No documents found in the database.")