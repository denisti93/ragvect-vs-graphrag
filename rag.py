from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# Load PDF with metadata
loader = PyPDFLoader("processo.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print(f"Sample chunk length: {len(chunks[0].page_content)} characters")

embeddings = OpenAIEmbeddings()

# Add metadata to help with retrieval
for i, chunk in enumerate(chunks):
    if not chunk.metadata:
        chunk.metadata = {}
    chunk.metadata['chunk_id'] = i

vectorstore = FAISS.from_documents(chunks, embeddings)

# Configure retriever with better parameters
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 5
    }
)

# Custom prompt for better answers
prompt_template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Use the context provided to give accurate and detailed answers.

Context:
{context}

Question: {question}

Answer in Portuguese:"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["context", "question"]
)

# Create QA chain with custom prompt
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    ),
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)

# Interactive Q&A loop
print("\n" + "="*60)
print("RAG System Ready! Ask questions about the document.")
print("Type 'quit', 'exit', or 'sair' to stop.")
print("="*60 + "\n")

while True:
    question = input("Your question: ").strip()
    
    if not question:
        continue
    
    # Check for exit commands
    if question.lower() in ['quit', 'exit', 'sair', 'q']:
        print("\nGoodbye!")
        break
    
    try:
        result = qa.invoke({"query": question})
        
        print(f"\nAnswer: {result['result']}")
        print(f"Source documents used: {len(result.get('source_documents', []))}")
        if result.get('source_documents'):
            print(f"First source (page {result['source_documents'][0].metadata.get('page', 'N/A')})")
        print("\n" + "-"*60 + "\n")
    except Exception as e:
        print(f"\nError: {e}\n")