from langchain_core.messages import HumanMessage,SystemMessage
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.document_loaders import JSONLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from nltk.tokenize import word_tokenize
import nltk

nltk.download('punkt_tab')


model = OllamaLLM(model="llama3.1", temperature=0.0)

embeddings = OllamaEmbeddings(model="llama3.1")


# you are loading db using path here
new_db = FAISS.load_local("data/chroma_33", embeddings, allow_dangerous_deserialization=True)


results_with_scores = new_db.similarity_search("The song is about anger, frustration and desperation.", k=100)

for doc in results_with_scores:
    print(f"Score: {doc.page_content}")


retriever = BM25Retriever.from_documents(results_with_scores, preprocess_func=word_tokenize)

result = retriever.invoke("anger, frustration and desperation")

print(result[0])

