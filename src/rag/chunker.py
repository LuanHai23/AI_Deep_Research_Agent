from langchain_text_splitters import CharacterTextSplitter

def chunk_papers(full_text: str):
    # Khởi tạo các công cụ cắt chuỗi văn bản với các tham số tùy chỉnh
    text_splitter = CharacterTextSplitter(
        separator=" ", # Ép cắt cứng theo đúng số lượng ký tự, không quan tâm dấu câu
        chunk_size = 500, # Kích thước mỗi chunk tầm 500 ký tự
        chunk_overlap = 200, # Cho phép chunk chép lại 200 ký tự để giữ ngữ cảnh
        length_function = len, # Dùng để đếm độ dài đếm theo ký tự
    )
    # Thực hiện cắt chuỗi văn bản thành các đoạn nhỏ hơn
    chunks = text_splitter.split_text(full_text)
    print(f"Document split into {len(chunks)} chunks.")

    for i, chunk in enumerate(chunks[:2]):
        print(f"Chunk {i+1} (Size: {len(chunk)} characters):\n{chunk}\n{'-'*40}")
        print(chunk + "...")
    return chunks

if __name__ == "__main__":
    mock_paper_content = (
        "AI agents are autonomous entities directed by AI models to achieve specific goals. "
        "Unlike standard LLMs that just answer questions, agents can use tools, browse the web, "
        "and execute code. " * 50 
    )
    results = chunk_papers(mock_paper_content)