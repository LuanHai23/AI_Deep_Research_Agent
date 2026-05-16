# from src.rag.embedder import model, util
# from src.tools.search import search_academic_papers
import chromadb

chroma_client = chromadb.PersistentClient(path="data/vector_store") # Đường dẫn đến thư mục lưu trữ vector database
collection = chroma_client.get_or_create_collection(name="academic_papers")
def save_to_db(records: list[dict]):
    print(f"Saving {len(records)} records to the database...")
    documents = [record['abstract'] for record in records]
    ids = [record["id"] for record in records]
    embeddings = [record["vector"] for record in records]
    metadatas = [record["metadata"] for record in records]
    
    # Đẩy nó vào chromaDB
    collection.add(
        documents=documents,
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas
    )
    print("Records saved successfully.")

def search_db(query_vector: list[float], top_k: int = 5):
    print("Searching the database...")
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k # lấy top_k kết quả gần nhất, có thể điều chỉnh tùy theo nhu cầu
    )
    print(f"Search completed. Found {len(results['ids'][0])} results.")
    return results

# collection.add(
#     documents = [
#         "this is a document about pineapple",
#         "This is a document about oranges"
#     ],
#     ids=["id1", "id2"]
# )
# results = collection.query(
#     query_texts=["This is a query document about hawaiian"],
#     n_results=1, # Số lượng kết quả trả về, có thể điều chỉnh tùy theo nhu cầu
#     where_document= {"$contains": "oranges"}, # Điều kiện lọc tài liệu, ví dụ: chỉ tìm kiếm trong các tài liệu chứa từ "oranges"
# )
# pprint(results)