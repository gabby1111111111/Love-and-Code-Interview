"""
Story Node Manager for Game Loop Pacing

Triggers story nodes based on spaced repetition progress (forgetting curve).
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum


class ReviewBox(int, Enum):
    """Review box levels for spaced repetition."""
    NEW = 0
    BOX_1 = 1
    BOX_2 = 2
    BOX_3 = 3
    BOX_4 = 4
    BOX_5 = 5


@dataclass
class StoryTrigger:
    """Represents a story trigger condition."""
    box_level: int
    triggered: bool = False
    description: str = ""


class StoryManager:
    """
    Manages story nodes based on forgetting curve progression.
    
    Story nodes trigger when:
    - First question reaches Box 1 (遗忘曲线第一阶段)
    - First question reaches Box 3 (中期巩固)
    - Overall mastery reaches certain threshold
    """
    
    def __init__(self):
        self.answered_questions = 0
        self.correct_answers = 0
        self.total_questions = 0
        
        # Track story triggers based on box progression
        self.triggers = {
            "box_1_first": StoryTrigger(
                box_level=1,
                description="首次征服遗忘曲线第一阶段 - 基因手术/初次觉醒"
            ),
            "box_3_first": StoryTrigger(
                box_level=3,
                description="知识巩固 - 晋升试炼"
            ),
            "mastery_70": StoryTrigger(
                box_level=-1,  # Special: based on overall stats
                description="70%掌握率 - 荣誉时刻"
            )
        }
        
        # Separate test mode for Astartes script
        self.test_mode_enabled = False
        self.test_mode_step = 0
    
    def check_box_milestone(self, question_box_levels: List[int]) -> Optional[str]:
        """
        Check if any box milestone triggers a story node.
        
        Args:
            question_box_levels: List of all questions' current box levels
            
        Returns:
            Story trigger key if triggered, None otherwise
        """
        # Check Box 1 milestone
        if not self.triggers["box_1_first"].triggered:
            if any(box >= 1 for box in question_box_levels):
                self.triggers["box_1_first"].triggered = True
                return "box_1_first"
        
        # Check Box 3 milestone
        if not self.triggers["box_3_first"].triggered:
            if any(box >= 3 for box in question_box_levels):
                self.triggers["box_3_first"].triggered = True
                return "box_3_first"
        
        # Check mastery threshold
        if not self.triggers["mastery_70"].triggered:
            if self.get_mastery_rate() >= 0.7 and self.total_questions >= 5:
                self.triggers["mastery_70"].triggered = True
                return "mastery_70"
        
        return None
    
    def record_answer(self, is_correct: bool):
        """Record answer result."""
        self.answered_questions += 1
        if is_correct:
            self.correct_answers += 1
        self.total_questions += 1
    
    def get_success_rate(self) -> float:
        """Calculate success rate."""
        if self.total_questions == 0:
            return 0.0
        return self.correct_answers / self.total_questions
    
    def get_mastery_rate(self) -> float:
        """Calculate mastery rate (questions in box 2+)."""
        # This should be calculated from actual question stats
        return self.get_success_rate()  # Simplified for now
    
    # === Test Mode (Astartes Script) ===
    def enable_test_mode(self):
        """Enable the Astartes test script (separate from main flow)."""
        self.test_mode_enabled = True
        self.test_mode_step = 0
    
    def get_test_mode_trigger(self) -> Optional[str]:
        """Get test mode trigger (Astartes script only)."""
        if not self.test_mode_enabled:
            return None
        
        if self.test_mode_step == 3:
            return "test_gene_surgery"
        elif self.test_mode_step == 6:
            return "test_promotion"
        
        return None
    
    def advance_test_step(self):
        """Advance test mode step."""
        if self.test_mode_enabled:
            self.test_mode_step += 1
