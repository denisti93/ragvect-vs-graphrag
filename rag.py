from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

loader = PyPDFLoader("processo.pdf")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)

print(len(chunks))

embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever()

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4o-mini"), 
    chain_type="stuff", 
    retriever=retriever
)

result = qa.run("What is the main topic of the document?")
print(result)