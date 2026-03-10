from pydantic import BaseModel
from typing import List, Optional

class ChatChunk(BaseModel):
    """
    Represents a chunk of conversation from a SillyTavern JSONL file.
    """
    text: str
    character_name: str
    chat_file: str
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    world_line: Optional[str] = None
    parent_chunk_id: Optional[str] = None
