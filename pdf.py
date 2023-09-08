import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import (
    FAISS,
)  # Boa para testar localmente, não é tao robusta quando pycone
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

if __name__ == "__main__":
    pdf_path = "data\ArtigoCafe.pdf"
    loader = PyPDFLoader(pdf_path)

    documents = loader.load()

    text_splitter = CharacterTextSplitter(
        chunk_size=1000, chunk_overlap=30, separator="\n"
    )

    docs = text_splitter.split_documents(documents=documents)

    embedding = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(docs, embedding=embedding)

    vector_store.save_local("faiss_index_react")

    new_vector_store = FAISS.load_local("faiss_index_react", embeddings=embedding)

    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(), chain_type="stuff", retriever=new_vector_store.as_retriever()
    )
    res = qa.run("Me de o contexto geral do projeto")
    print(res)

    # Resposta para a query:
    #  O projeto tem três etapas principais: pré-processamento,
    #  treinamento de uma rede de convolução e implementação de um sistema multi-estágio
    #  de classificação de doenças de folhas de café.
    # O protótipo desenvolvido tem um custo muito baixo, pois a placa,
    #  a câmera e o display custam US$ 30 e a caixa 3D custa cerca de US$ 5.
    #  Além disso, o modelo alcança resultados comparáveis ​​ou melhores do que projetos embutidos em smartphones
    # e computado
    # res de placa única, como o Jetson Nano e o Raspberry Pi.
