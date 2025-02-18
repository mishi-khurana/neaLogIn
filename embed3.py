from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
import numpy as np
import json

# Define metadata function to update each song's metadata
def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["song"] = record.get("song")
    metadata["song_id"] = record.get("song_id")
    return metadata

# Initialize the model and embedding
model = OllamaLLM(model="llama3.1", temperature=0.0)
embedding = OllamaEmbeddings(model="llama3.1")
persist_directory = 'data/chroma_3words/'

# Load your data (songs) from the JSON file
loader = JSONLoader(
    file_path='./data/songs.json',
    jq_schema='.[]',
    content_key="lyrics",
    metadata_func=metadata_func
)

data = loader.load()

# Generate summary and genre metadata for each song
def generateData(data):
    for d in data:
        summary = model.invoke([
            SystemMessage(content="Summarise the lyrics of this song in a list of 5 words in this format: Sad, Regret, Betrayal. The output should only contain the five words")
        ] + [
            HumanMessage(content=d.page_content)
        ]).strip()

        genre = model.invoke([
            SystemMessage(content="Extract a list of up to 5 genres of this song in this format: Pop, Rock, Folk. The output should only contain the five words")
        ] + [
            HumanMessage(content=d.page_content)
        ]).strip()

        song_id = d.metadata.get("song_id")

        # Update metadata with summary and genre
        d.metadata.update({
            "summary": summary,
            "genre": genre,
            "song_id": song_id
        })

        print(f"Processed data for song: {d.metadata['song_id']}")

# Function to store embeddings in the vector database
def store_embeddings(data):
    for d in data:
        # Create structured text for embedding
        embedding_text = f"Genre: {d.metadata['genre']}. Keywords: {d.metadata['summary']}. Song ID: {d.metadata['song_id']}"
        d.page_content = embedding_text  # Replace lyrics with structured text for embedding

    # Store the documents with embeddings in Chroma
    vectordb = Chroma.from_documents(documents=data, embedding=embedding, persist_directory=persist_directory)
    vectordb.persist()  # Save the vector database
    return vectordb

# Run the functions to generate metadata and store embeddings
generateData(data)
vectordb = store_embeddings(data)

# Querying the vector database and retrieving all documents
documents = vectordb._collection.get_all()

# Print the first document's metadata and embedding
if documents:
    first_doc = documents[0]  # Access the first document
    print("First Document Metadata:", first_doc.metadata)
    print("First Document Embedding:", first_doc.embedding)
else:
    print("No documents found in the database.")
