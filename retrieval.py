'''from langchain_core.messages import AIMessage, HumanMessage
from langchain_ollama import ChatOllama,OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma


from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from shared import get_words

def get_data(user_input):

    model = OllamaLLM(model="llama3.1", temperature=0.0)
    embedding = OllamaEmbeddings(model="llama3.1")

    db_dir = 'data/chroma/'

    vectordb = Chroma(
     persist_directory=db_dir,
     embedding_function=embedding
    )



    metadata_field_info = [
    AttributeInfo(
        name="song_id",
        description="song id",
        type="string",
    ),
    AttributeInfo(
        name="title",
        description="Name of song",
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
    words = get_words()

 
    question = {f"You are an expert lyricist. A user has requested a list of songs which lyrics are, or are linked to the words {words}. Can you provide a list of songs that would be suitable for this user?"}
    print (question)

    res = retriever.invoke(question)

    res

    for r in res:
     print(r.metadata.get("song_id"))
    
    return res'''