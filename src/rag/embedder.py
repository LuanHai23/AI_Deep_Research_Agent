import sentence_transformers
from sentence_transformers import SentenceTransformer, util
import numpy as np
# Mô hình embedder sử dụng SentenceTransformer để tạo embeddings cho các đoạn văn bản
Embedder_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    print("Generating embeddings for chunks...")
    # Sử dụng encode() để tạo embeddings cho từng chunk văn bản
    # Thông số conver_to_tensor = False để trả về numpy array thay vì tensor PyTorch
    embeddings = Embedder_model.encode(chunks, show_progress_bar=True)
    print(f"Generated embeddings for {len(chunks)} chunks.")
    embeddings_list = embeddings.tolist()

    print("Embeddings generated successfully.")
    return embeddings_list

if __name__ == "__main__":
    mock_chunks = [
        "AI agents are autonomous entities directed by AI models to achieve specific goals.",
        "Unlike standard LLMs that just answer questions, agents can use tools, browse the web, and execute code."
    ]
    embeddings = generate_embeddings(mock_chunks)
    print(f"Embeddings for mock chunks: {embeddings}")
    print(f"Embedding dimension: {len(embeddings[0])}")



# sentences = ["This is an example sentence", "Each sentence is converted", "This is an example sentence", "This is an example sentence"]
# embeddings = model.encode(sentences)
# print(embeddings)

# similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1])
# print(f"Similarity: {similarity.item()}")
# similarity2 = util.pytorch_cos_sim(embeddings[2], embeddings[3])
# print(f"Similarity with itself: {similarity2.item()}")
# similarity3 = util.pytorch_cos_sim(embeddings[0], embeddings[2])
# print(f"Similarity with itself: {similarity3.item()}")