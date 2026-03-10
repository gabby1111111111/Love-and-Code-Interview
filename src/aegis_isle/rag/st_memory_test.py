import json
import os
import sys

def parse_st_chat_log(file_path, chunk_size=6):
    """
    解析 SillyTavern 的 jsonl 聊天记录文件，并将对话按 chunk_size 划分为块。
    不包含系统提示和额外的表格数据，只提取真实对话。
    """
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return []

    messages = []
    
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                # 跳过第一行的 metadata
                if "chat_metadata" in data:
                    continue
                
                # 提取真正的对话内容
                if "mes" in data and "is_user" in data:
                    is_user = data["is_user"]
                    name = data.get("name", "User" if is_user else "Char")
                    content = data["mes"]
                    
                    messages.append({
                        "role": "user" if is_user else "char",
                        "name": name,
                        "content": content
                    })
            except json.JSONDecodeError:
                continue

    print(f"Total conversational messages extracted: {len(messages)}")
    
    # 切分 Chunk
    chunks = []
    current_chunk = []
    current_text = ""
    
    for idx, msg in enumerate(messages):
        current_chunk.append(msg)
        current_text += f"{msg['name']}: {msg['content']}\n\n"
        
        # 满足 chunk_size 或者是最后一条消息
        if len(current_chunk) >= chunk_size or idx == len(messages) - 1:
            chunks.append({
                "chunk_index": len(chunks),
                "messages_count": len(current_chunk),
                "text_content": current_text.strip()
            })
            current_chunk = []
            current_text = ""

    print(f"Total chunks created: {len(chunks)}")
    return chunks

if __name__ == "__main__":
    file_path = "E:\\SillyTaven\\SillyTavern\\data\\default-user\\chats\\邹峥1\\邹峥 - 2026-02-27@17h23m32s.jsonl"
    
    chunks = parse_st_chat_log(file_path)
    
    # 打印前 3 个 Chunk 看看效果
    print("\n--- SAMPLE CHUNKS ---")
    for chunk in chunks[:3]:
        print(f"\n[Chunk #{chunk['chunk_index']} (Messages: {chunk['messages_count']})]")
        # 打印部分内容预览，保留换行符结构
        content_lines = chunk['text_content'].split('\n')
        preview = '\n'.join(content_lines[:15])
        if len(content_lines) > 15:
            preview += "\n... (truncated)"
        print(preview)
        print("-" * 60)
