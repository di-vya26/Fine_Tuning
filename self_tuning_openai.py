import json
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import JSONLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.docstore.document import Document
import os


os.environ["OPENAI_API_KEY"] = "sk-..."  # Replace with your actual key

# --- Step 1: Load JSON data and convert to documents ---

def load_json_to_docs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    docs = []
    for entry in data:
        content = json.dumps(entry, indent=2)  # flatten to string
        docs.append(Document(page_content=content, metadata={}))
    return docs

docs = load_json_to_docs("your_data.json")  # Your JSON file

# --- Step 2: Chunk and embed documents ---

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)

embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embedding_model)

# --- Step 3: Define the Prompt Generator LLM Chain ---

llm = ChatOpenAI(model="gpt-4")

prompt_template = PromptTemplate(
    input_variables=["query"],
    template="""You are a helpful assistant that rewrites user questions into better search queries for retrieving relevant documents.

User query: "{query}"

Improved search query:"""
)

prompt_generator = LLMChain(llm=llm, prompt=prompt_template)

# --- Step 4: Retrieve relevant documents using generated prompt ---

def self_tuning_rag(user_query):
    print(f"\nUser Query: {user_query}")
    search_prompt = prompt_generator.run(user_query)
    print(f"Generated Search Prompt: {search_prompt}")
    
    retrieved_docs = vectorstore.similarity_search(search_prompt, k=5)
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 5}),
        chain_type="stuff"
    )
    
    answer = qa_chain.run(user_query)
    return answer

# --- Step 5: Ask your RAG system a question ---

response = self_tuning_rag("Why did the Hadoop job fail on node X?")
print("\nAnswer:", response)
