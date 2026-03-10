import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from aegis_isle.rag.embedder import TextEmbedder

logger = logging.getLogger(__name__)

class DailyDigest:
    """
    统一日记聚合器 (DailyDigest)
    将 LifeEventBus 中的散落 JSONL 事件流，在每日某个时刻聚合成一篇结构化的 Markdown 日记，
    并将该日记通过 BGE-zh 模型进行 embedding，写入专门的 diary/ FAISS 索引中。
    """
    
    def __init__(self, events_dir: str = "data/diary/events", digests_dir: str = "data/diary/digests", vectorstore_dir: str = "data/vectorstore/diary"):
        self.events_dir = events_dir
        self.digests_dir = digests_dir
        self.vectorstore_dir = vectorstore_dir
        os.makedirs(self.events_dir, exist_ok=True)
        os.makedirs(self.digests_dir, exist_ok=True)
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        
        # 复用 STMemoryManager 的 embedder，避免重复加载 PyTorch 模型消耗大量内存
        from aegis_isle.rag.st_memory_manager import memory_manager
        self.embedder = memory_manager.embedder
        
        self.index_path = os.path.join(self.vectorstore_dir, "daily_diary.index")

    def _read_jsonl(self, filename: str) -> List[Dict]:
        filepath = os.path.join(self.events_dir, filename)
        if not os.path.exists(filepath):
            return []
        
        events = []
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))
        except Exception as e:
            logger.error(f"Failed to read {filename}: {e}")
        return events
        
    def _clear_jsonl(self, filename: str):
        """编译完成后清理已处理的事件"""
        filepath = os.path.join(self.events_dir, filename)
        if os.path.exists(filepath):
            try:
                # 简单清空文件内容（在实际高并发系统中需要文件锁或 rotate 机制）
                open(filepath, 'w').close()
            except Exception as e:
                logger.error(f"Failed to clear {filename}: {e}")

    def collect_events(self) -> str:
        """读取当前的 JSONL 事件，生成并返回 Markdown 格式的日记正文"""
        browsing_events = self._read_jsonl("browsing.jsonl")
        interview_events = self._read_jsonl("interview.jsonl")
        chat_events = self._read_jsonl("chat_summary.jsonl")
        char_events = self._read_jsonl("character_activity.jsonl")
        
        if not any([browsing_events, interview_events, chat_events, char_events]):
            return ""
            
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        sections = []
        sections.append(f"# {today_date} 的记忆与日记\n")
        
        if browsing_events:
            sections.append("## 📱 今日浏览足迹")
            for ev in browsing_events:
                action = ev.get('action', '')
                title = ev.get('title', '')
                platform = ev.get('platform', '')
                tags = " ".join([f"#{t}" for t in ev.get('tags', [])])
                item = f"- [{action}] {platform}《{title}》 {tags}"
                if ev.get('comment'):
                    item += f" \n  > 评论了: \"{ev.get('comment')}\""
                sections.append(item)
            sections.append("")
                
        if interview_events:
            sections.append("## 📝 面试练习记录")
            correct = sum(1 for e in interview_events if e.get('verdict') == 'correct')
            total = len(interview_events)
            sections.append(f"- 今日共练习 {total} 道题目，正确 {correct} 题 (正确率: {int(correct/total*100) if total else 0}%)")
            for ev in interview_events:
                q = ev.get('question', '')[:30].replace('\n', ' ')
                res = "✅" if ev.get('verdict') == "correct" else "❌"
                sections.append(f"  - {res} [{ev.get('category')}] {q}...")
            sections.append("")
                
        if chat_events:
            sections.append("## 💬 角色互动摘要")
            for ev in chat_events:
                summary = ev.get('summary', '')
                char = ev.get('character', 'Unknown')
                sections.append(f"- [{char}] {summary}")
            sections.append("")
            
        if char_events:
            sections.append("## 💭 自主意识日志 (内心独白)")
            for ev in char_events:
                char = ev.get('character', 'Unknown')
                details = ev.get('details', {})
                topic = details.get('source_topic', '')
                reaction = details.get('char_reaction', '')
                sections.append(f"### {char} 的精神角落")
                sections.append(f"> 触发事件：{topic}")
                sections.append(f"{reaction}")
            sections.append("")
            
        return "\n".join(sections)

    async def compile_and_index(self):
        """生成日记文本 → BGE-zh 嵌入 → 追加到 diary FAISS"""
        diary_content = self.collect_events()
        if not diary_content:
            logger.info("今日暂无事件需要编译日记。")
            return {"status": "skipped", "message": "无事件"}
            
        date_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        
        # 1. 存为 markdown 文件归档
        md_file = os.path.join(self.digests_dir, f"{date_str}.md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write(diary_content)
            
        # 2. 存入 FAISS
        doc = Document(
            page_content=diary_content,
            metadata={"date": date_str, "type": "daily_diary"}
        )
        def _index():
            if os.path.exists(self.index_path):
                vectorstore = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                vectorstore.add_documents([doc])
                vectorstore.save_local(self.index_path)
                logger.info(f"Appended daily diary to existing FAISS index: {self.index_path}")
            else:
                vectorstore = FAISS.from_documents([doc], self.embedder)
                vectorstore.save_local(self.index_path)
                logger.info(f"Created new diary FAISS index at: {self.index_path}")
                
        import asyncio
        await asyncio.to_thread(_index)
            
        # 3. 清理已编译的 JSONL 事件
        for src in ["browsing.jsonl", "interview.jsonl", "chat_summary.jsonl", "character_activity.jsonl"]:
            self._clear_jsonl(src)
            
        return {"status": "success", "file": md_file}

    async def search(self, query: str, k: int = 3) -> str:
        """语义检索日记，返回格式化字符串 (由于是统一日记，不用区分角色)"""
        if not os.path.exists(self.index_path):
            return ""
            
        def _search() -> str:
            try:
                vectorstore = FAISS.load_local(self.index_path, self.embedder, allow_dangerous_deserialization=True)
                docs = vectorstore.similarity_search(query, k=k)
                if not docs:
                    return ""
                    
                context = "【过往日记回忆】\n"
                for d in docs:
                    date = d.metadata.get("date", "未知时间")
                    context += f"--- {date} ---\n{d.page_content}\n"
                    
                return context
            except Exception as e:
                logger.error(f"Failed to search daily digest: {e}")
                return ""
                
        import asyncio
        return await asyncio.to_thread(_search)

# 单例提供检索用
daily_digest = DailyDigest()
