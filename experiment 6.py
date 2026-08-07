import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# 1. Knowledge Base

documents = [
    """
    Generative Artificial Intelligence is a branch of AI that creates
    new content such as text, images, audio, video and computer programs.
    """,

    """
    Large Language Models are transformer-based models trained on massive
    text datasets. They are used for text generation, summarization,
    translation, question answering and conversational AI.
    """,

    """
    Retrieval-Augmented Generation combines information retrieval with
    text generation. It retrieves relevant documents from an external
    knowledge base and gives them to a language model as context.
    """,

    """
    Vector databases store high-dimensional embeddings and perform
    similarity searches. Examples include FAISS, ChromaDB,
    Pinecone, Weaviate and Milvus.
    """,

    """
    Prompt engineering is the process of designing clear instructions
    that guide a language model to produce accurate and useful responses.
    Common techniques include zero-shot, few-shot and role-based prompting.
    """,

    """
    Fine-tuning adapts a pretrained language model to a specific domain
    or task by training it further using a smaller domain-specific dataset.
    """
]

# 2. Load Embedding Model

print("Loading embedding model...")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

# 3. Convert Documents to Embeddings

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True
).astype("float32")

# Normalize vectors
faiss.normalize_L2(document_embeddings)

# 4. Create FAISS Vector Database

dimension = document_embeddings.shape[1]

vector_database = faiss.IndexFlatIP(dimension)

vector_database.add(document_embeddings)

# 5. Load Text Generation Model

print("Loading FLAN-T5 model...")

generator = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    framework="pt"
)

# 6. Retrieve Documents

def retrieve_documents(query, top_k=2):

    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    similarity_scores, document_indices = vector_database.search(
        query_embedding,
        top_k
    )

    retrieved_documents = []

    for index, score in zip(document_indices[0], similarity_scores[0]):
        retrieved_documents.append({
            "document": documents[index].strip(),
            "score": float(score)
        })

    return retrieved_documents

# 7. Generate Answer

def generate_answer(query, retrieved_documents):

    context = "\n\n".join(
        item["document"] for item in retrieved_documents
    )

    prompt = f"""
Answer the question using only the information below.

Context:
{context}

Question:
{query}

Instructions:
1. Give a clear answer.
2. Use only the context.
3. If the answer is unavailable, reply:
"The answer is not available in the knowledge base."

Answer:
"""

    result = generator(
        prompt,
        max_new_tokens=150,
        do_sample=False
    )

    return result[0]["generated_text"]

# 8. Main Program

print("\n")
print("="*60)
print("RETRIEVAL-AUGMENTED GENERATION (RAG)")
print("="*60)

while True:

    query = input("\nEnter your question (or type 'exit'): ")

    if query.lower() == "exit":
        print("\nThank you!")
        break

    retrieved = retrieve_documents(query, top_k=2)

    answer = generate_answer(query, retrieved)

    print("\nRetrieved Documents")
    print("-"*60)

    for i, item in enumerate(retrieved, start=1):
        print(f"\nDocument {i}")
        print(item["document"])
        print(f"Similarity Score : {item['score']:.4f}")

    print("\nGenerated Answer")
    print("-"*60)
    print(answer)

    print("\n" + "="*60)
