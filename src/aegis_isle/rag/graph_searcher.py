import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class BaseGraphSearcher(ABC):
    """
    面向接口的图谱检索器基类。
    预留此接口，以便未来可以零代码侵入地切换到 Neo4j / LlamaIndex Property Graph / MS GraphRAG。
    """
    
    @abstractmethod
    async def search(self, query: str, universe_id: str, character_name: str) -> str:
        """
        根据 query、宇宙 ID 和角色名进行图谱检索，返回可直接注入 prompt 的格式化字符串。
        """
        pass

class JsonlGraphSearcher(BaseGraphSearcher):
    """
    当前的临时实现模式：将预处理生成的 graph_nodes.jsonl 和 graph_edges.jsonl 加载到内存。
    基于关键词或正则进行属性查询。
    """
    def __init__(self, data_dir: str = "debug/chunks"):
        self.data_dir = Path(data_dir)
        # 内部缓存： universe_id -> {"nodes": [...], "edges": [...]}
        self._cache: Dict[str, Dict[str, List[Dict]]] = {}
        
    def _load_universe_graph(self, universe_id: str):
        if universe_id in self._cache:
            return
            
        nodes = []
        edges = []
        
        # 寻找匹配的 graph_nodes 和 graph_edges (不关心时间戳前缀，只认 universe_id 即可)
        # 由于实际文件名如: 买裙子_邹峥___2026_01_30_04h13m03s_20260306_211622_graph_nodes.jsonl
        # 我们遍历寻找包含 universe_id 的文件
        for f in self.data_dir.glob(f"*{universe_id}*_*_graph_nodes.jsonl"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip(): nodes.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error loading nodes {f}: {e}")

        for f in self.data_dir.glob(f"*{universe_id}*_*_graph_edges.jsonl"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip(): edges.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error loading edges {f}: {e}")
                
        self._cache[universe_id] = {"nodes": nodes, "edges": edges}
        logger.info(f"[GraphSearcher] 宇宙 {universe_id} - 已加载 {len(nodes)} 节点, {len(edges)} 边")

    async def search(self, query: str, universe_id: str, character_name: str) -> str:
        if not self.data_dir.exists():
            return ""
            
        safe_world = "".join([c for c in universe_id if c.isalnum() or c in (' ', '-', '_')]).strip()
        self._load_universe_graph(safe_world)
        
        graph_data = self._cache.get(safe_world)
        if not graph_data or (not graph_data["nodes"] and not graph_data["edges"]):
            return ""
            
        nodes = graph_data["nodes"]
        edges = graph_data["edges"]
        
        # 极简查询逻辑：如果带有目标角色名，提取它的属性和边
        target_node = None
        for n in nodes:
            if n.get("name") == character_name:
                target_node = n
                break
                
        if not target_node:
            # 没找到主角节点，返回空
            return ""
            
        # 抽取属性
        attrs = target_node.get("attributes", {})
        attr_str = " | ".join([f"{k}: {v}" for k, v in attrs.items()])
        
        # 抽取与他相关的边 (只取前 3 条最重要的)
        related_edges = [e for e in edges if e.get("source") == target_node.get("node_id") or e.get("target") == target_node.get("node_id")]
        edge_strs = []
        for e in related_edges[:3]:
            rel = e.get('relation', '未知')
            sent = e.get('sentiment', '')
            edge_strs.append(f"-> 关系:{rel} (情感:{sent})")
            
        context = f"【图谱雷达: 角色状态】\n属性: {attr_str}\n"
        if edge_strs:
            context += "交互边:\n" + "\n".join(edge_strs)
            
        return context

# 单例实例
graph_searcher = JsonlGraphSearcher()
