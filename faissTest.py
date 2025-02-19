from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
# from langchain.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import JSONLoader


def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["song"] = record.get("song")
    metadata["song_id"] = record.get("song_id")
    return metadata


model = OllamaLLM(model="llama3.1", temperature=0.0)
embedding = OllamaEmbeddings(model="llama3.1")

loader = JSONLoader(
    file_path='./data/songs.json',
    jq_schema='.[]',
    content_key="lyrics",
    metadata_func=metadata_func
    )

docs = loader.load()
print("Verifying fields", docs[0])  # Print the first record to verify fields
faiss_db_path = "data/test_DB"
print("1")
db = FAISS.from_documents(docs, embedding)
print("2")

new_db = FAISS.load_local("data/test_DB", embedding, allow_dangerous_deserialization=True)
print("3")

for doc in docs: 
    summary = model.invoke([
            SystemMessage(content="Summarise the lyrics of this song in a list of 5 words in this format: Sad, Regret, Betrayal. The output should only contain the five words")
        ] + [
            HumanMessage(content=doc.page_content)
        ])
    summary = summary.content.strip()  # Extract text

    genre = model.invoke([
            SystemMessage(content="Extract a list of up to 5 genres of this song in this format: Pop, Rock, Folk. The output should only contain the five words")
        ] + [
            HumanMessage(content=doc.page_content)
        ]).strip()
    genre = genre.content.strip()  # Extract text


    song_id = doc.metadata.get("song_id")
    
    doc.metadata.update({
        "summary": summary,
        "genre": genre,
        "song_id": song_id
        })

    print(f"data is + {doc.metadata}")


faiss_db_path = "data/test_DB"

# Save the FAISS index to disk
db.save_local(faiss_db_path)