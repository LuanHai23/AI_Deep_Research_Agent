from src.db.vector_store import search_db
from src.rag.embedder import generate_embeddings

def retrieve_relevant_chunks(user_query: str, top_k: int = 5):
    print(f"Retrieving relevant chunks for query: '{user_query}'")
    # Bước 1: Tạo embedding cho câu hỏi của người dùng
    # Vì generate_embeddings trả về một list các embedding, nên chúng ta lấy phần tử đầu tiên [0] để có được embedding của câu hỏi
    # ký hiệu [0] là để lấy embedding đầu tiên trong list, vì chúng ta chỉ tạo embedding cho một câu hỏi duy nhất
    query_vector = generate_embeddings([user_query])[0]

    # Bước 2: Tìm kiếm trong vector database để lấy các chunk có embedding gần nhất với câu hỏi
    search_results = search_db(query_vector=query_vector, top_k=top_k)

    # Bước 3: Lọc lấy phần text từ kết quả của ChromaDB
    # Chroma trả về dữ liệu nằm trong list của list
    retrieved_texts = search_results['documents'][0]
    retrieved_metadatas = search_results['metadatas'][0]

    print(f"Retrieved {len(retrieved_texts)} relevant chunks with metadata: {retrieved_metadatas}")
    print(f"Retrieved {len(retrieved_texts)} relevant chunks.")
    return retrieved_texts, retrieved_metadatas