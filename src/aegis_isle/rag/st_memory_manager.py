import os
import json
import logging
import asyncio
from typing import List, Dict, Any, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from aegis_isle.rag.st_memory import ChatChunk
from aegis_isle.rag.embedder import TextEmbedder

logger = logging.getLogger(__name__)

class STMemoryManager:
    """
    Manages the ingestion, storage, and retrieval of SillyTavern chat logs using FAISS.
    """
    
    def __init__(self, vectorstore_dir: str = "data/vectorstore/st_memory"):
        self.vectorstore_dir = vectorstore_dir
        os.makedirs(self.vectorstore_dir, exist_ok=True)
        # Reuse existing TextEmbedder
        self.text_embedder = TextEmbedder()
        # FAISS integration in Langchain usually needs an object with an `embed_documents` and `embed_query` method.
        # We will wrap our TextEmbedder to provide the interface expected by Langchain FAISS
        class EmbedderWrapper:
            def __init__(self, ae_embedder):
                self.embedder = ae_embedder
            def embed_documents(self, texts: List[str]) -> List[List[float]]:
                # Our embed_texts is async, we need to adapt it since FAISS.from_documents is sync
                # But Langchain FAISS can also take an embedding function.
                raise NotImplementedError("Use async FAISS or sync wrapper for document embedding")
            def embed_query(self, query: str) -> List[float]:
                import asyncio
                return asyncio.run(self.embedder.embed_query(query))
                
        # To avoid fighting sync/async FAISS ingest right now, we can use the async-friendly pattern
        # or load a standard sentence-transformers wrapper for ingestion compatibility if needed.
        from langchain_community.embeddings import HuggingFaceEmbeddings
        self.embedder = HuggingFaceEmbeddings(model_name=self.text_embedder.model_name)
        
        self.indices: Dict[str, FAISS] = {}
        
    def _get_index_path(self, character_name: str, world_line: Optional[str] = None) -> str:
        import re
        # Allow alphanumeric, Chinese characters, spaces, hyphens, and underscores
        safe_name = re.sub(r'[^\w\u4e00-\u9fff \-_]', '', character_name).strip()
        filename = f"{safe_name}.index"
        if world_line:
            safe_world = re.sub(r'[^\w\u4e00-\u9fff \-_]', '', world_line).strip()
            filename = f"{safe_name}_{safe_world}.index"
        
        # fallback for chinese character encoding issues in powershell args
        if not filename or filename.startswith("_") or filename == ".index":
            filename = "default_char.index"
            
        return os.path.join(self.vectorstore_dir, filename)

    def load_index(self, character_name: str, world_line: Optional[str] = None) -> Optional[FAISS]:
        """Loads an existing FAISS index for a character into memory."""
        index_key = f"{character_name}_{world_line}" if world_line else character_name
        
        if index_key in self.indices:
            return self.indices[index_key]
            
        index_path = self._get_index_path(character_name, world_line)
        if os.path.exists(index_path):
            try:
                # FAISS expects a directory containing index.faiss and index.pkl
                # Workaround for FAISS Unicode path crash on Windows:
                import shutil
                temp_path = os.path.join(os.path.dirname(index_path), "temp_index_load")
                if os.path.exists(temp_path):
                    shutil.rmtree(temp_path)
                shutil.copytree(index_path, temp_path)
                
                vectorstore = FAISS.load_local(temp_path, self.embedder, allow_dangerous_deserialization=True)
                
                try:
                    shutil.rmtree(temp_path)
                except Exception:
                    pass
                
                self.indices[index_key] = vectorstore
                logger.info(f"Loaded ST memory index for {character_name}")
                return vectorstore
            except Exception as e:
                logger.error(f"Failed to load FAISS index from {index_path}: {e}")
        
        return None

    def ingest_chunks(self, chunks: List[ChatChunk], character_name: str, world_line: Optional[str] = None):
        """
        Ingests a list of ChatChunk objects into a FAISS index, saving it to disk.
        """
        if not chunks:
            logger.warning("No chunks to ingest.")
            return

        documents = []
        for chunk in chunks:
            metadata = chunk.model_dump()
            # Remove the actual text from metadata to avoid duplication
            text = metadata.pop("text") 
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
            
        index_key = f"{character_name}_{world_line}" if world_line else character_name
        vectorstore = self.load_index(character_name, world_line)
        
        if vectorstore:
            # Append to existing index
            vectorstore.add_documents(documents)
            logger.info(f"Added {len(documents)} chunks to existing index for {character_name}")
        else:
            # Create new index
            vectorstore = FAISS.from_documents(documents, self.embedder)
            self.indices[index_key] = vectorstore
            logger.info(f"Created new index for {character_name} with {len(documents)} chunks")
            
        # Save to disk (Workaround for FAISS C++ writer crash on Windows Unicode paths)
        import shutil
        index_path = self._get_index_path(character_name, world_line)
        temp_path = os.path.join(os.path.dirname(index_path), "temp_index_save")
        
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        vectorstore.save_local(temp_path)
        
        if os.path.exists(index_path):
            shutil.rmtree(index_path)
        os.rename(temp_path, index_path)
        
        logger.info(f"Saved ST memory index to {index_path} via temp workaround")

        # --- 将 chat summary 写进 LifeEventBus ---
        try:
            from aegis_isle.rag.event_logger import event_bus
            import asyncio
            
            # Simple summarization based on count for now, DailyDigest will extract content
            summary = f"记录了 {len(documents)} 段对话记忆。"
            
            # Since ingest_chunks is running in a sync executor but maybe within an async loop,
            # we need to be careful about event loop submission.
            try:
                loop = asyncio.get_running_loop()
                asyncio.run_coroutine_threadsafe(
                    event_bus.log_chat_summary(world_line or "default", character_name, summary),
                    loop
                )
            except RuntimeError:
                # No running event loop in this thread
                asyncio.run(event_bus.log_chat_summary(world_line or "default", character_name, summary))
                
        except Exception as e:
            logger.error(f"Failed to log chat_summary to event bus: {e}", exc_info=True)
        # ---------------------------------------------

    # ── 元数据预过滤相关常量 ──
    LOCATION_KEYWORDS = {
        "酒吧": ["酒吧", "bar", "锚点"],
        "餐厅": ["餐厅", "餐馆", "饭店", "吃饭", "法餐", "Le Rêve"],
        "大学": ["大学", "学院", "教室", "研讨室", "办公室"],
        "家": ["家里", "公寓", "卧室", "书房", "客厅"],
    }
    
    TIME_KEYWORDS = {
        "一月": "01月", "二月": "02月", "三月": "03月", "四月": "04月",
        "五月": "05月", "六月": "06月", "七月": "07月", "八月": "08月",
        "九月": "09月", "十月": "10月", "十一月": "11月", "十二月": "12月",
        "1月": "01月", "2月": "02月", "3月": "03月", "4月": "04月",
        "5月": "05月", "6月": "06月", "7月": "07月", "8月": "08月",
        "9月": "09月", "10月": "10月", "11月": "11月", "12月": "12月",
    }
    
    def _extract_location_hints(self, query: str) -> List[str]:
        """从 query 中提取地点和时间关键词，返回 hint 列表"""
        query_lower = query.lower()
        hints = []
        # 地点关键词
        for _category, keywords in self.LOCATION_KEYWORDS.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    hints.extend(keywords)  # 把同类的所有别名都加进去
                    break
        # 时间关键词
        for trigger, month_pattern in self.TIME_KEYWORDS.items():
            if trigger in query_lower:
                hints.append(month_pattern)
        return list(set(hints))
    
    def _fetch_scene_meta(self, parent_chunk_id: str, chat_file: str) -> dict:
        """根据 parent_chunk_id 从 parent_chunks.jsonl 读取 scene_meta"""
        import glob
        prefix = chat_file.replace("_sub_chunks.jsonl", "")
        search_pattern = os.path.join("debug", "chunks", f"{prefix}_parent_chunks.jsonl")
        files = glob.glob(search_pattern)
        if not files:
            files = glob.glob(os.path.join("debug", "chunks", "*_parent_chunks.jsonl"))
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if parent_chunk_id in line:
                            data = json.loads(line)
                            if data.get("parent_chunk_id") == parent_chunk_id:
                                return data.get("scene_meta", {})
            except Exception:
                pass
        return {}
    
    def _post_filter_by_metadata(self, docs: List[Document], hints: List[str]) -> List[Document]:
        """
        对 FAISS 返回的 docs 按元数据匹配 hints 排序。
        匹配的排在前面，不匹配的排在后面（fallback 补位）。
        """
        matched = []
        unmatched = []
        for doc in docs:
            parent_id = doc.metadata.get("parent_chunk_id", "")
            chat_file = doc.metadata.get("chat_file", "")
            scene_meta = self._fetch_scene_meta(parent_id, chat_file)
            location = scene_meta.get("location", "")
            date = scene_meta.get("date", "")
            meta_text = f"{location} {date}".lower()
            
            if any(h.lower() in meta_text for h in hints):
                matched.append(doc)
            else:
                unmatched.append(doc)
        
        logger.info(f"[Memory] 元数据过滤: hints={hints}, matched={len(matched)}, unmatched={len(unmatched)}")
        return matched + unmatched

    async def search_memory(self, query: str, character_name: str, world_line: Optional[str] = None, k: int = 4) -> List[Document]:
        """
        Searches the FAISS index for a character and returns relevant chunks.
        Supports multi-universe search if world_line is comma-separated.
        1. 提取 query 中的地点/时间关键词
        2. 从多个 FAISS 库并发获取 候选
        3. 聚合、按分数去重
        4. 按 scene_meta 元数据优先过滤
        5. 返回 top-k
        """
        world_lines = []
        if world_line:
            world_lines = [w.strip() for w in world_line.split(",") if w.strip()]
        if not world_lines:
            world_lines = [None]  # 回退到单基准宇宙
            
        vectorstores = []
        for wl in world_lines:
            vs = self.load_index(character_name, wl)
            if vs: vectorstores.append(vs)
            
        if not vectorstores:
            return []
            
        hints = self._extract_location_hints(query)
        
        # 多宇宙联合查询
        all_results_with_scores = []
        
        def _search_vs(vs):
            # FAISS returns (Document, score), lower score is better (L2 distance)
            return vs.similarity_search_with_score(query, k=k * 4)
            
        # 并发跑 FAISS 查询 (FAISS search is blocking, run in thread)
        loop = asyncio.get_event_loop()
        tasks = [loop.run_in_executor(None, _search_vs, vs) for vs in vectorstores]
        results_lists = await asyncio.gather(*tasks)
        
        for res_list in results_lists:
            all_results_with_scores.extend(res_list)
            
        # 按分数排序 (L2 distance 越小越好)
        all_results_with_scores.sort(key=lambda x: x[1])
        
        # 去重并提取 Document
        seen_texts = set()
        raw_results = []
        for doc, score in all_results_with_scores:
            if doc.page_content not in seen_texts:
                seen_texts.add(doc.page_content)
                raw_results.append(doc)
                if len(raw_results) >= k * 4:  # 只保留聚合后的 top k*4 去做后过滤
                    break
                    
        if hints:
            filtered = self._post_filter_by_metadata(raw_results, hints)
            return filtered[:k]
        
        return raw_results[:k]

    def _fetch_episode_plot(self, parent_chunk_id: str, chat_file: str) -> str:
        """从 episodes.jsonl 中查找对应的 plot 摘要"""
        import glob, json, re
        
        # parent_chunk_id 格式: xxx_chunk_004 → 提取数字 4
        match = re.search(r'_chunk_0*(\d+)$', parent_chunk_id)
        if not match:
            return ""
        chunk_num_int = int(match.group(1))
        
        # 使用模糊匹配找到对应的 episodes.jsonl (因为真实文件名可能带有时间戳)
        # 例如: 16岁被收养私生活乱___2026_01_20_10h49m  -> 匹配 *16岁被收养私生活乱___2026_01_20_10h49m*_episodes.jsonl
        world_line = parent_chunk_id.split("_chunk_")[0]
        ep_files = glob.glob(os.path.join("debug", "chunks", f"*{world_line}*_episodes.jsonl"))
        
        if not ep_files:
            return ""
            
        for fpath in ep_files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        data = json.loads(line)
                        ep_id = data.get("episode_id", "")
                        # ep_id 格式可能是 ep_xxx_004，提取末尾数字
                        ep_match = re.search(r'_0*(\d+)$', ep_id)
                        if ep_match and int(ep_match.group(1)) == chunk_num_int:
                            return data.get("plot", "")
            except Exception as e:
                logger.error(f"Error reading episode file {fpath}: {e}")
        return ""

    def _fetch_parent_chunk_text(self, parent_chunk_id: str, chat_file: str, sub_chunk_text: str = "") -> str:
        """获取 parent chunk 的上下文，以命中的 sub-chunk 位置为中心截取"""
        if not parent_chunk_id:
            return ""
        import glob, json
        
        WINDOW_SIZE = 300
        
        prefix = chat_file.replace("_sub_chunks.jsonl", "")
        search_pattern = os.path.join("debug", "chunks", f"{prefix}_parent_chunks.jsonl")
        
        files = glob.glob(search_pattern)
        if not files:
            files = glob.glob(os.path.join("debug", "chunks", "*_parent_chunks.jsonl"))
            
        for fpath in files:
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        if parent_chunk_id in line:
                            data = json.loads(line)
                            if data.get("parent_chunk_id") == parent_chunk_id:
                                meta = data.get("scene_meta", {})
                                user_msg = data.get("user_msg", "")
                                meta_str = " | ".join(f"{k}: {v}" for k, v in meta.items() if v and isinstance(v, str))
                                full_text = data.get("full_ai_text", "")
                                
                                # ── 以 sub-chunk 命中位置为中心截取 ──
                                if sub_chunk_text and sub_chunk_text in full_text:
                                    hit_pos = full_text.index(sub_chunk_text)
                                    center = hit_pos + len(sub_chunk_text) // 2
                                    half = WINDOW_SIZE // 2
                                    start = max(0, center - half)
                                    end = min(len(full_text), center + half)
                                    # 边界自适应：哪边不够就从另一边多取
                                    if start == 0:
                                        end = min(len(full_text), WINDOW_SIZE)
                                    elif end == len(full_text):
                                        start = max(0, len(full_text) - WINDOW_SIZE)
                                    snippet = full_text[start:end]
                                    # 添加省略标记
                                    if start > 0:
                                        snippet = "…" + snippet
                                    if end < len(full_text):
                                        snippet = snippet + "…"
                                    text_preview = snippet
                                elif len(full_text) > WINDOW_SIZE:
                                    text_preview = full_text[:WINDOW_SIZE] + "…（场景略）"
                                else:
                                    text_preview = full_text
                                
                                # ── 获取 episode plot 摘要 ──
                                plot = self._fetch_episode_plot(parent_chunk_id, chat_file)
                                
                                parts = [f"[场景元数据: {meta_str}]"]
                                if plot:
                                    parts.append(f"[场景摘要]: {plot}")
                                parts.append(f"[User曾说]: {user_msg}")
                                parts.append(f"[相关上下文]: {text_preview}")
                                return "\n".join(parts)
            except Exception:
                pass
        return ""

    def format_context_for_prompt(self, documents: List[Document], max_chunks: int = 3) -> str:
        """
        Formats retrieved documents into a string suitable for LLM injection.
        Clearly frames content as HISTORICAL background, not current user message.
        """
        if not documents:
            return ""
            
        formatted_chunks = []
        seen_parents = set()
        
        for i, doc in enumerate(documents, 1):
            if len(formatted_chunks) >= max_chunks:
                break
                
            chat_file = doc.metadata.get("chat_file", "历史聊天")
            parent_id = doc.metadata.get("parent_chunk_id")
            
            # 去重逻辑：如果已经注入过这个父块，跳过
            if parent_id and parent_id in seen_parents:
                continue
                
            # 尝试获取完整的 Parent Chunk（传入 sub-chunk 原文用于居中截取）
            full_context = self._fetch_parent_chunk_text(parent_id, chat_file, sub_chunk_text=doc.page_content) if parent_id else ""
            
            # 记录已处理的父块
            if full_context and parent_id:
                seen_parents.add(parent_id)
            
            # 如果没找到， fallback 回原本的小 chunk
            content = full_context if full_context else doc.page_content
            
            # 提取所属的宇宙/世界线
            universe = doc.metadata.get("world_line")
            universe_label = universe if universe else "基准宇宙"
            
            formatted_chunks.append(f"【记忆片段 {len(formatted_chunks) + 1}】[所属宇宙: {universe_label}]（来源：{chat_file}）\n{content}")
            
        context_string = "\n\n".join(formatted_chunks)
        return (
            "══════════ 长期记忆（历史对话节选）══════════\n"
            "下方是你与用户之间真实发生过的历史对话片段，仅供你参考以保证记忆连贯性。\n"
            "注意：这些是【历史背景信息】，不是用户当前说的话。\n"
            "请在阅读完后，继续响应用户最新的消息。\n\n"
            f"{context_string}\n"
            "══════════════════════════════════════════"
        )

# Global singleton instance
memory_manager = STMemoryManager()
