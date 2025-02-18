from langchain_core.messages import AIMessage, HumanMessage,SystemMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import PyPDFLoader, TextLoader, JSONLoader
from langchain_community.vectorstores import Chroma
import numpy as np

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo

#  jq_schema='to_entries | map({page_content: .lyrics, metadata: (.value | {title} )})',



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
print(data[0])  # Print the first record to verify fields

data[300].page_content

# data[1].page_content = "hello"


# model = ChatOllama(model="llama3.1:8b")

model = OllamaLLM(model="llama3.1", temperature=0.0)

for d in data:
    d.page_content = model.invoke([SystemMessage(content="Give me sentiment of song in one word only in plain text!")]+
                                  [HumanMessage(content=d.page_content)])
    print(d.page_content)


res = model.invoke([SystemMessage(content="You are an expert lyricist, what is sentiment of this in one line:")]+[HumanMessage(content=data[5].page_content)])

res
# model = OllamaLLM(model)

embedding = OllamaEmbeddings(model="llama3.1")

len(data)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 5,
    chunk_overlap = 0
)
docs = []
splits = text_splitter.split_documents(data)

len(splits)

persist_directory = 'data/chroma_1line/'
vectordb = Chroma.from_documents(documents=data, embedding=embedding, persist_directory =persist_directory)


word1 = "happy"
word2 = "love"
word3 = "dance"
query = f"The song is about {word1}, {word2} and {word3}"  # Proper text format
query_embedding = embedding.embed_query(query)  # Convert to vector
# query_embedding = np.array(query_embedding)
# query_embedding /= np.linalg.norm(query_embedding)  
print("Query embedding:", query_embedding[:20])


'''res = retriever.invoke(question)'''

#res = vectordb.similarity_search("Sad", k=1)

#res[0]
#res[2]

#res[0].metadata['song']
#res[0].page_content


res = vectordb.max_marginal_relevance_search_by_vector(
        query_embedding, 
        k=1,  # Number of results
        fetch_k=100,  # Number of results to fetch
        lambda_mult=0.5  # 0.5 balances similarity & diversity
    )

res

# res = vectordb.similarity_search(question,k=3)

len(res)
res[2].metadata['song']
res[2].page_content


metadata_field_info = [
    AttributeInfo(
        name="song_id",
        description="song id",
        type="string",
    ),
    AttributeInfo(
        name="name",
        description="The is name of the song",
        type="string",
    ),
]

document_content_description = "Lyrics of song"

retriever = SelfQueryRetriever.from_llm(
    model,
    vectordb,
    document_content_description,
    metadata_field_info,
    verbose=True
)


# question = "romantic"

# res = retriever.invoke(question)



for r in res:
    print(r.metadata)