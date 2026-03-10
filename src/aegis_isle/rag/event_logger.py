import os
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

class LifeEventBus:
    """
    统一事件记录器 (The Unified Diary System - Event Sourcing)
    实时将各端（ST-Companion-Link, Love&Code, Aegis Chat）的行为写入 JSONL 尾部。
    后续由 DailyDigest 定时提取并编译为日记。
    """
    
    def __init__(self, base_dir: str = "data/diary/events"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.files = {
            "browsing": self.base_dir / "browsing.jsonl",
            "interview": self.base_dir / "interview.jsonl",
            "chat_summary": self.base_dir / "chat_summary.jsonl",
            "character_activity": self.base_dir / "character_activity.jsonl"
        }
        
        # 确保文件存在
        for path in self.files.values():
            if not path.exists():
                path.touch()

    def _append_to_log(self, log_type: str, data: dict):
        """线程安全的追加写入"""
        filepath = self.files.get(log_type)
        if not filepath:
            logger.error(f"未知的日志类型: {log_type}")
            return
            
        data["timestamp"] = datetime.now().isoformat()
        
        try:
            # 简单追加写入，因为是通过 FastAPI 单实例队列或异步处理，冲突概率低，且 jsonl 本身就是 append
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
            logger.debug(f"已记录事件 [{log_type}]: {data.get('action', 'N/A')}")
        except Exception as e:
            logger.error(f"写入事件日志失败 [{log_type}]: {e}", exc_info=True)

    # 异步包装，方便在 FastAPI 中使用而不阻塞主线程
    async def log_browsing(self, action: str, title: str, tags: list, url: str, platform: str, comment: str = None):
        """记录用户的浏览行为 (CL)"""
        data = {
            "action": action,
            "title": title,
            "tags": tags,
            "url": url,
            "platform": platform,
            "comment": comment
        }
        # 使用 asyncio.to_thread 避免密集 IO 阻塞事件循环 (虽然此处 IO 极小)
        await asyncio.to_thread(self._append_to_log, "browsing", data)

    async def log_interview(self, question_text: str, correct: bool, category: str, tags: list):
        """记录用户的面试练习行为 (Love & Code)"""
        data = {
            "action": "answer_question",
            "question": question_text,
            "verdict": "correct" if correct else "incorrect",
            "category": category,
            "tags": tags
        }
        await asyncio.to_thread(self._append_to_log, "interview", data)

    async def log_chat_summary(self, universe_id: str, character: str, summary: str):
        """记录角色互动的剧集摘要 (Aegis Chat ingest)"""
        data = {
            "action": "chat_episode",
            "universe_id": universe_id,
            "character": character,
            "summary": summary
        }
        await asyncio.to_thread(self._append_to_log, "chat_summary", data)

    async def log_character_activity(self, universe_id: str, character: str, action_type: str, details: dict):
        """记录角色自主做的事情（CharLifeAgent）"""
        data = {
            "action": action_type,
            "universe_id": universe_id,
            "character": character,
            "details": details
        }
        await asyncio.to_thread(self._append_to_log, "character_activity", data)

# 全局单例
event_bus = LifeEventBus()
