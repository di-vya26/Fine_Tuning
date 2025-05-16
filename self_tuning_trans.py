from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType
from langchain.vectorstores import Milvus
from langchain.embeddings import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import pipeline, AutoTokenizer
import torch
import json
import os

# ---------------- Step 1: Load JSON and convert to Documents ----------------
def load_json_to_docs(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    docs = []
    for entry in data:
        content = json.dumps(entry, indent=2)
        docs.append(Document(page_content=content, metadata={}))
    return docs

docs = load_json_to_docs("your_data.json")  # JSON file path

# ---------------- Step 2: Split text into chunks ----------------
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(docs)
texts = [doc.page_content for doc in chunks]

# ---------------- Step 3: Connect to Milvus ----------------
connections.connect("default", host="localhost", port="19530")

# ---------------- Step 4: Create Milvus Collection ----------------
collection_name = "rag_data"

if not Collection.exists(collection_name):
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, description="RAG Milvus Collection")
    collection = Collection(name=collection_name, schema=schema)
else:
    collection = Collection(name=collection_name)

collection.load()

# ---------------- Step 5: Embed and Insert into Milvus ----------------
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embed_model.encode(texts, show_progress_bar=True)

entities = [
    texts,
    [vec.tolist() for vec in embeddings]
]
collection.insert(entities)
collection.flush()

# ---------------- Step 6: Setup Prompt Generator LLM ----------------
generator = pipeline("text-generation", model="tiiuae/falcon-7b-instruct",
                     tokenizer=AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct"),
                     device=0 if torch.cuda.is_available() else -1,
                     max_new_tokens=100)

def generate_prompt(user_query):
    prompt = f"Rewrite the following user query to make it a better search prompt for document retrieval:\n\nUser query: {user_query}\n\nSearch prompt:"
    result = generator(prompt, do_sample=True, top_k=50)[0]["generated_text"]
    return result.split("Search prompt:")[-1].strip()

# ---------------- Step 7: Retrieve and Answer ----------------
def retrieve_and_answer(user_query):
    print(f"\nOriginal Query: {user_query}")
    search_prompt = generate_prompt(user_query)
    print(f"Generated Search Prompt: {search_prompt}")

    # Embed the search prompt
    query_vec = embed_model.encode([search_prompt])[0].tolist()

    collection.load()
    results = collection.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=3,
        output_fields=["text"]
    )

    context = "\n\n".join([hit.entity.get("text") for hit in results[0]])

    answer_prompt = f"""Answer the following user question based on the provided context.\n\nContext:\n{context}\n\nQuestion: {user_query}\n\nAnswer:"""

    response = generator(answer_prompt, do_sample=True, top_k=50)[0]["generated_text"]
    return response.split("Answer:")[-1].strip()

# ---------------- Step 8: Test the system ----------------
response = retrieve_and_answer("Why did the Hadoop job fail on node X?")
print("\nAnswer:", response)