import os

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain import VectorDBQA, OpenAI
import pinecone
from langchain.chains import RetrievalQA

pinecone.init(api_key="914a0f4d-fe87-46d5-8ee8-abaddf1155da", environment="gcp-starter")

if __name__ == "__main__":
    loader = TextLoader("data\medium_blog.txt", encoding="utf-8")
    document = loader.load()
    print(document)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    # Adiciona ao vector DB
    docsearch = Pinecone.from_documents(
        texts, embedding=embeddings, index_name="medium-blogs-embeddings-index"
    )
    # Pegar os embeddings pertos do prompt, transformar de novo para chunks e adicionar contexto
    # Usamos stuffing para adicionar o contexto
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,  # Mostra aonde a LLM pegou as respostas
    )

    query = "What is a vector database? Give me a 15 word answer for a begginer."

    result = qa({"query": query})

    print(result)
