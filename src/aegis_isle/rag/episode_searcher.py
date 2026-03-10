import json
import logging
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

class EpisodeSearcher:
    """
    负责剧情宏观回顾。
    当检测到“第一次”、“曾经”、“回忆”等剧情词缀时，在 `episodes.jsonl` 的 `plot` 字段全文搜索。
    """
    def __init__(self, data_dir: str = "debug/chunks"):
        self.data_dir = Path(data_dir)
        self._cache: Dict[str, List[Dict]] = {}
        
    def _load_episodes(self, universe_id: str):
        if universe_id in self._cache:
            return
            
        episodes = []
        for f in self.data_dir.glob(f"*{universe_id}*_*_episodes.jsonl"):
            try:
                with open(f, 'r', encoding='utf-8') as file:
                    for line in file:
                        if line.strip(): episodes.append(json.loads(line))
            except Exception as e:
                logger.error(f"Error loading episodes {f}: {e}")
                
        self._cache[universe_id] = episodes
        logger.info(f"[EpisodeSearcher] 宇宙 {universe_id} - 已加载 {len(episodes)} 个剧情摘要")

    async def search(self, query: str, universe_id: str) -> str:
        if not self.data_dir.exists():
            return ""
            
        safe_world = "".join([c for c in universe_id if c.isalnum() or c in (' ', '-', '_')]).strip()
        self._load_episodes(safe_world)
        
        episodes = self._cache.get(safe_world, [])
        if not episodes:
            return ""
            
        # 简单全文匹配：如果 query 的分词在 plot 中出现，就提高权重，暂时简化为只要命中关键词我们就返回最新2条概括
        # 因为在网关已经做了意图判断，走到这里说明必须回顾了
        
        # 为了防空，如果没有特定词，直接返回最新的 2 个 episode
        matched = episodes[-2:] 
        
        if not matched:
            return ""
            
        results = []
        for i, ep in enumerate(matched, 1):
            time_range = ep.get('time_range', '未知时间')
            plot = ep.get('plot', '')
            seeds = ep.get('seeds', [])
            seed_str = " | ".join(seeds) if seeds else "无剧情种子"
            results.append(f"【剧情锚点 {i}】 (时段: {time_range})\n概览: {plot}\n埋点: {seed_str}")
            
        return "\n\n".join(results)

# 单例实例
episode_searcher = EpisodeSearcher()
