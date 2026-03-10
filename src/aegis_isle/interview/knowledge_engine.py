"""
Knowledge Engine for Interview Prep System

Implements spaced repetition learning algorithm for interview questions.
Manages question database and integrates with LLM for content generation.
"""

import json
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from enum import IntEnum

from pydantic import BaseModel, Field, validator

from ..core.logging import logger
from ..core.config import settings


class Difficulty(IntEnum):
    """Question difficulty levels."""
    VERY_EASY = 1
    EASY = 2
    MEDIUM = 3
    HARD = 4
    VERY_HARD = 5


class ReviewBox(IntEnum):
    """Spaced repetition review boxes (0 = new, 1-5 = increasing intervals)."""
    NEW = 0
    BOX_1 = 1  # 1 day
    BOX_2 = 2  # 3 days
    BOX_3 = 3  # 7 days
    BOX_4 = 4  # 14 days
    BOX_5 = 5  # 30 days


class Question(BaseModel):
    """
    Interview question data model with spaced repetition support.

    Attributes:
        id: Unique question identifier
        content: Question text
        answer_key: Optional reference answer/key points
        difficulty: Difficulty level (1-5)
        review_box: Current spaced repetition box (0-5)
        next_review: DateTime string for next review (ISO format)
        created_at: Creation timestamp
        category: Question category (e.g., "algorithms", "system_design")
        tags: Associated tags for filtering
        source: Source of the question (e.g., job description, study material)
        attempts: Number of times question was attempted
        correct_answers: Number of correct answers
    """

    id: str = Field(..., description="Unique question identifier")
    content: str = Field(..., min_length=10, description="Question text")
    answer_key: Optional[str] = Field(None, description="Reference answer or key points")
    gabriella_context: Optional[str] = Field(None, description="Gabriella's Cyber Tea Party explanation")
    pro_context: Optional[str] = Field(None, description="Professional interviewer perspective")
    difficulty: int = Field(..., ge=1, le=5, description="Difficulty level (1-5)")
    review_box: int = Field(default=0, ge=0, le=5, description="Spaced repetition box (0-5)")
    next_review: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Next review datetime (ISO format)"
    )
    created_at: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Creation timestamp"
    )
    category: str = Field(default="general", description="Question category")
    tags: List[str] = Field(default_factory=list, description="Associated tags")
    source: str = Field(default="unknown", description="Source of question")
    attempts: int = Field(default=0, ge=0, description="Number of attempts")
    correct_answers: int = Field(default=0, ge=0, description="Number of correct answers")

    @validator('next_review', 'created_at')
    def validate_datetime_string(cls, v):
        """Validate datetime string format."""
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("DateTime must be in ISO format")

    @property
    def next_review_datetime(self) -> datetime:
        """Get next_review as datetime object."""
        return datetime.fromisoformat(self.next_review.replace('Z', '+00:00'))

    @property
    def created_at_datetime(self) -> datetime:
        """Get created_at as datetime object."""
        return datetime.fromisoformat(self.created_at.replace('Z', '+00:00'))

    @property
    def success_rate(self) -> float:
        """Calculate success rate (0.0 to 1.0)."""
        if self.attempts == 0:
            return 0.0
        return self.correct_answers / self.attempts

    def is_due_for_review(self) -> bool:
        """Check if question is due for review."""
        return datetime.utcnow() >= self.next_review_datetime

    def update_review_schedule(self, is_correct: bool):
        """
        Update review schedule based on answer correctness.

        Args:
            is_correct: Whether the answer was correct
        """
        self.attempts += 1

        if is_correct:
            self.correct_answers += 1
            # Move to next box (increase interval)
            if self.review_box < ReviewBox.BOX_5:
                self.review_box += 1

            # Calculate next review time based on box
            intervals = {
                ReviewBox.BOX_1: timedelta(days=1),
                ReviewBox.BOX_2: timedelta(days=3),
                ReviewBox.BOX_3: timedelta(days=7),
                ReviewBox.BOX_4: timedelta(days=14),
                ReviewBox.BOX_5: timedelta(days=30)
            }

            next_interval = intervals.get(self.review_box, timedelta(days=1))
            self.next_review = (datetime.utcnow() + next_interval).isoformat()
        else:
            # Reset to immediate review
            self.review_box = ReviewBox.NEW
            self.next_review = datetime.utcnow().isoformat()


class KnowledgeEngine:
    """
    Manages interview questions database with spaced repetition learning.

    Features:
    - JSON-based question database
    - LLM-powered question generation from text/job descriptions
    - Spaced repetition algorithm for optimal learning
    - Progress tracking and analytics
    """

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize KnowledgeEngine.

        Args:
            db_path: Path to question database JSON file
        """
        self.db_path = db_path or Path("data/interview_db.json")
        self.questions: Dict[str, Question] = {}
        self.load_database()

        logger.info(f"KnowledgeEngine initialized with {len(self.questions)} questions")

    def load_database(self):
        """Load questions from JSON database file."""
        try:
            if self.db_path.exists():
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                # Convert dict data to Question objects
                for q_id, q_data in data.get('questions', {}).items():
                    try:
                        question = Question(**q_data)
                        self.questions[q_id] = question
                    except Exception as e:
                        logger.warning(f"Failed to load question {q_id}: {e}")

                logger.info(f"Loaded {len(self.questions)} questions from database")
            else:
                # Create empty database
                self.save_database()
                logger.info("Created new question database")

        except Exception as e:
            logger.error(f"Failed to load question database: {e}")
            self.questions = {}

    def save_database(self):
        """Save questions to JSON database file."""
        try:
            # Ensure directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # Convert questions to serializable format
            data = {
                'questions': {q_id: q.dict() for q_id, q in self.questions.items()},
                'metadata': {
                    'total_questions': len(self.questions),
                    'last_updated': datetime.utcnow().isoformat()
                }
            }

            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            logger.debug(f"Saved {len(self.questions)} questions to database")

        except Exception as e:
            logger.error(f"Failed to save question database: {e}")

    async def ingest_data(self, text: str, jd_context: Optional[str] = None, use_cache: bool = True, language: str = "zh") -> List[Question]:
        """
        Generate interview questions from text using LLM with chunking.
        Supports parallel processing and caching.

        Args:
            text: Source text (study material, documentation, etc.)
            jd_context: Optional job description for contextual questions
            use_cache: Whether to use cached questions if available
            language: Language code for questions (zh/en/jp)

        Returns:
            List of generated Question objects

        Raises:
            Exception: If LLM generation fails
        """
        try:
            # Check cache first
            import hashlib
            content_hash = hashlib.md5((text + (jd_context or "")).encode()).hexdigest()
            cache_path = Path(f"data/ingestion_cache_{content_hash}.json")
            
            if use_cache and cache_path.exists():
                logger.info(f"📦 Found cached ingestion data: {cache_path}")
                try:
                    with open(cache_path, 'r', encoding='utf-8') as f:
                        cached_data = json.load(f)
                        questions = [Question(**q) for q in cached_data]
                        
                    # Add to database
                    for question in questions:
                        self.questions[question.id] = question
                    self.save_database()
                    
                    logger.info(f"✅ Loaded {len(questions)} questions from cache")
                    return questions
                except Exception as e:
                    logger.warning(f"Failed to load cache: {e}, proceeding with generation")

            from ..rag.generator import LLMGenerator, GenerationConfig
            from langchain.text_splitter import RecursiveCharacterTextSplitter

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            chunks = text_splitter.split_text(text)
            total_chunks = len(chunks)
            
            logger.info(f"📚 Splitting text into {total_chunks} chunks for processing...")

            # Initialize text generator
            config = GenerationConfig(
                model=settings.default_llm_model,
                max_tokens=2000,
                temperature=0.7
            )
            generator = LLMGenerator(config, provider=settings.llm_provider)

            # Process chunks in parallel with semaphore
            semaphore = asyncio.Semaphore(10) # Limit concurrency to 10
            
            async def process_chunk(i, chunk):
                async with semaphore:
                    logger.info(f"🔄 Processing Chunk {i}/{total_chunks}...")
                    try:
                        # Build prompt for this chunk
                        prompt = self._build_question_generation_prompt(chunk, jd_context, language)
                        
                        # Generate questions for this chunk
                        result = await generator.generate(prompt)
                        
                        # Parse questions
                        chunk_questions = self._parse_generated_questions(result.generated_text, jd_context)
                        
                        logger.info(f"✅ Processed Chunk {i}/{total_chunks}, found {len(chunk_questions)} questions")
                        return chunk_questions
                    except Exception as e:
                        logger.error(f"❌ Failed to process chunk {i}: {e}")
                        return []

            # Create tasks
            tasks = [process_chunk(i, chunk) for i, chunk in enumerate(chunks, 1)]
            
            # Run parallel execution
            results = await asyncio.gather(*tasks)
            
            # Flatten results
            all_questions = [q for sublist in results for q in sublist]

            # Deduplicate questions
            logger.info(f"🔍 Deduplicating {len(all_questions)} questions...")
            deduplicated = self._deduplicate_questions(all_questions)
            logger.info(f"✨ Final count: {len(deduplicated)} unique questions")

            # Save to cache
            try:
                with open(cache_path, 'w', encoding='utf-8') as f:
                    json.dump([q.dict() for q in deduplicated], f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved ingestion cache to {cache_path}")
            except Exception as e:
                logger.warning(f"Failed to save cache: {e}")

            # Add to database
            for question in deduplicated:
                self.questions[question.id] = question

            self.save_database()
            logger.info(f"💾 Saved {len(deduplicated)} questions to database")

            return deduplicated

        except Exception as e:
            logger.error(f"Failed to ingest data: {e}")
            raise

    def _deduplicate_questions(self, questions: List[Question]) -> List[Question]:
        """Remove duplicate questions based on content similarity."""
        unique_questions = []
        seen_contents = set()
        
        for q in questions:
            # Normalize content for comparison
            normalized = q.content.lower().strip()
            
            # Check for exact duplicates
            if normalized not in seen_contents:
                seen_contents.add(normalized)
                unique_questions.append(q)
            else:
                logger.debug(f"Skipping duplicate: {q.content[:50]}...")
        
        return unique_questions

    def _build_question_generation_prompt(self, text: str, jd_context: Optional[str] = None, language: str = "zh") -> str:
        """Build prompt for LLM question generation."""
        
        lang_instructions = {
            "zh": {
                "instruction": "请用简体中文生成问题和答案",
                "example_q": "什么是二分查找的时间复杂度？",
                "example_a": "O(log n) - 因为每次迭代会排除一半的搜索空间",
                "guidelines": """指导方针：
1. 难度级别：1=非常简单，2=简单，3=中等，4=困难，5=非常困难
2. 问题应具体且可回答
3. 包含多样化的难度级别（主要是2-4级）
4. 提供简洁但准确的answer_key
5. 使用相关的categories和tags
6. 专注于实用知识和理解
7. 问题应符合实际面试场景"""
            },
            "en": {
                "instruction": "Generate questions and answers in English",
                "example_q": "What is the time complexity of binary search?",
                "example_a": "O(log n) - because we eliminate half the search space in each iteration",
                "guidelines": """Guidelines:
1. Difficulty scale: 1=Very Easy, 2=Easy, 3=Medium, 4=Hard, 5=Very Hard
2. Questions should be specific and answerable
3. Include diverse difficulty levels (mix of 2-4 mostly)
4. Provide concise but accurate answer_key
5. Use relevant categories and tags
6. Focus on practical knowledge and understanding
7. Make questions realistic for actual interviews"""
            }
        }
        
        lang_cfg = lang_instructions.get(language, lang_instructions["zh"])

        context_section = ""
        if jd_context:
            context_section = f"""

Job Description Context:
{jd_context[:1500]}

Focus questions on skills and requirements mentioned in this job description.
"""

        prompt = f"""{lang_cfg['instruction']}

Based on the following text, generate relevant interview questions that could be asked about this topic.

Source Text:
{text}
{context_section}

Please generate 5-10 interview questions in the following JSON format:

```json
{{
  "questions": [
    {{
      "content": "{lang_cfg['example_q']}",
      "answer_key": "{lang_cfg['example_a']}",
      "difficulty": 3,
      "category": "algorithms",
      "tags": ["binary_search", "time_complexity", "algorithms"]
    }}
  ]
}}
```

{lang_cfg['guidelines']}

Return only the JSON format, no additional text."""

        return prompt

    def _parse_generated_questions(self, llm_output: str, source_context: Optional[str] = None) -> List[Question]:
        """
        Parse LLM-generated questions from JSON output.

        Args:
            llm_output: Raw LLM response containing JSON
            source_context: Original source context for metadata

        Returns:
            List of Question objects
        """
        questions = []

        try:
            # Extract JSON from LLM output
            import re
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_output, re.DOTALL)
            if not json_match:
                # Try to find JSON without code blocks
                json_match = re.search(r'\{.*\}', llm_output, re.DOTALL)

            if not json_match:
                raise ValueError("No JSON found in LLM output")

            json_text = json_match.group(1) if json_match.lastindex else json_match.group(0)
            data = json.loads(json_text)

            # Parse questions from JSON
            for i, q_data in enumerate(data.get('questions', [])):
                try:
                    # Generate unique ID
                    question_id = f"gen_{datetime.utcnow().timestamp()}_{i:02d}"

                    # Create Question object
                    question = Question(
                        id=question_id,
                        content=q_data['content'],
                        answer_key=q_data.get('answer_key', ''),
                        difficulty=q_data.get('difficulty', 3),
                        category=q_data.get('category', 'general'),
                        tags=q_data.get('tags', []),
                        source=source_context or "llm_generated"
                    )

                    questions.append(question)

                except Exception as e:
                    logger.warning(f"Failed to parse question {i}: {e}")
                    continue

        except Exception as e:
            logger.error(f"Failed to parse LLM-generated questions: {e}")
        return questions

    def get_next_question(self, exclude_ids: List[str] = None, preferred_difficulty: int = None) -> Optional[Question]:
        """
        Get next question with comprehensive balancing.

        Balances three factors:
        1. Forgetting curve (遗忘曲线) - prioritize due questions
        2. Repetition limit - maximum 5 times per question
        3. Success rate - high success questions need less repetition

        Returns:
            Next Question object or None if no questions available
        """
        if not self.questions:
            return None

        exclude_ids = exclude_ids or []
        now = datetime.utcnow()

        # Filter out excluded questions
        available_questions = {
            qid: q for qid, q in self.questions.items()
            if qid not in exclude_ids
        }

        if not available_questions:
            logger.warning("All questions have been answered recently!")
            return None

        # Hard limit: 5 repetitions maximum
        MAX_REPETITIONS = 5

        # Get questions due for review
        due_questions = [
            q for q in available_questions.values()
            if q.next_review_datetime <= now and q.review_box > ReviewBox.NEW
        ]

        if due_questions:
            # Filter out questions that exceeded 5 repetitions
            filtered_due = [q for q in due_questions if q.attempts < MAX_REPETITIONS]
            
            if filtered_due:
                # Score each question based on multiple factors
                scored_questions = []
                for q in filtered_due:
                    score = self._calculate_question_priority(q, now)
                    scored_questions.append((score, q))
                
                # Sort by score (lower is higher priority)
                scored_questions.sort(key=lambda x: x[0])
                selected = scored_questions[0][1]
                
                logger.debug(
                    f"Selected review question (Box {selected.review_box}, "
                    f"attempt {selected.attempts + 1}/{MAX_REPETITIONS}, "
                    f"success rate: {selected.success_rate:.1%}): {selected.content[:50]}..."
                )
                return selected
            else:
                # All due questions exceeded 5 repetitions
                logger.warning("All due questions exceeded 5 repetitions, prioritizing new questions...")
                # Fall through to new questions

        # Get new questions (never reviewed)
        new_questions = [
            q for q in available_questions.values()
            if q.review_box == ReviewBox.NEW and q.attempts < MAX_REPETITIONS
        ]

        if new_questions:
            # Filter by preferred difficulty if specified
            if preferred_difficulty:
                difficulty_filtered = [q for q in new_questions if q.difficulty == preferred_difficulty]
                if difficulty_filtered:
                    new_questions = difficulty_filtered
            
            # Sort by: 1) difficulty (easy to hard), 2) fewer attempts, 3) creation date
            new_questions.sort(key=lambda q: (q.difficulty, q.attempts, q.created_at))
            logger.debug(f"Selected new question (attempt 1/{MAX_REPETITIONS}): {new_questions[0].content[:50]}...")
            return new_questions[0]

        # Last resort: if all questions exceeded limit, return None
        logger.info("No questions available - all exceeded 5 repetitions or are scheduled for future")
        return None
    
    def _calculate_question_priority(self, question: Question, current_time: datetime) -> float:
        """
        Calculate priority score for a question (lower = higher priority).
        
        Factors:
        1. Forgetting curve urgency (最重要)
        2. Repetition count penalty
        3. Success rate bonus (high success = less urgent)
        
        Returns:
            Priority score (lower is better)
        """
        # Factor 1: Urgency based on overdue time
        overdue_hours = (current_time - question.next_review_datetime).total_seconds() / 3600
        urgency_score = -overdue_hours  # Negative = more overdue = lower score = higher priority
        
        # Factor 2: Repetition penalty (more repetitions = higher score = lower priority)
        repetition_penalty = question.attempts * 10  # Each repetition adds 10 points
        
        # Factor 3: Success rate bonus
        # High success (>80%) = already mastered = lower priority
        # Low success (<50%) = needs practice = higher priority (but capped by repetition limit)
        success_rate = question.success_rate
        if success_rate >= 0.8:
            success_bonus = 20  # Well mastered, deprioritize
        elif success_rate >= 0.5:
            success_bonus = 0  # Moderate, neutral
        else:
            success_bonus = -15  # Struggling, prioritize (but not too much)
        
        # Combine factors
        # Urgency is most important (weight 1.0)
        # Repetition penalty is secondary (weight 1.0)
        # Success bonus is tertiary (weight 1.0)
        total_score = urgency_score + repetition_penalty + success_bonus
        
        return total_score

    def update_progress(self, question_id: str, is_correct: bool) -> bool:
        """
        Update question progress after answering.

        Args:
            question_id: ID of the answered question
            is_correct: Whether the answer was correct

        Returns:
            True if update successful, False if question not found
        """
        question = self.questions.get(question_id)
        if not question:
            logger.warning(f"Question not found for progress update: {question_id}")
            return False

        # Update review schedule
        question.update_review_schedule(is_correct)

        # Save database
        self.save_database()

        logger.info(
            f"Updated progress for question {question_id}: "
            f"{'correct' if is_correct else 'incorrect'}, "
            f"box={question.review_box}, "
            f"next_review={question.next_review_datetime.strftime('%Y-%m-%d %H:%M')}"
        )

        return True

    def get_questions_by_category(self, category: str) -> List[Question]:
        """Get all questions in a specific category."""
        return [q for q in self.questions.values() if q.category == category]

    def get_questions_by_difficulty(self, difficulty: int) -> List[Question]:
        """Get all questions of specific difficulty level."""
        return [q for q in self.questions.values() if q.difficulty == difficulty]

    def get_questions_due_for_review(self) -> List[Question]:
        """Get all questions currently due for review."""
        now = datetime.utcnow()
        return [q for q in self.questions.values() if q.next_review_datetime <= now]

    def get_progress_statistics(self) -> Dict[str, Any]:
        """
        Get learning progress statistics.

        Returns:
            Dictionary with various progress metrics
        """
        total_questions = len(self.questions)

        if total_questions == 0:
            return {
                'total_questions': 0,
                'questions_by_box': {},
                'due_for_review': 0,
                'overall_success_rate': 0.0,
                'questions_by_difficulty': {},
                'questions_by_category': {}
            }

        # Count questions by review box
        box_counts = {}
        for i in range(6):  # Boxes 0-5
            box_counts[f'box_{i}'] = len([q for q in self.questions.values() if q.review_box == i])

        # Count due questions
        due_count = len(self.get_questions_due_for_review())

        # Calculate overall success rate
        total_attempts = sum(q.attempts for q in self.questions.values())
        total_correct = sum(q.correct_answers for q in self.questions.values())
        overall_success_rate = (total_correct / total_attempts) if total_attempts > 0 else 0.0

        # Count by difficulty
        difficulty_counts = {}
        for i in range(1, 6):
            difficulty_counts[f'difficulty_{i}'] = len([q for q in self.questions.values() if q.difficulty == i])

        # Count by category
        categories = set(q.category for q in self.questions.values())
        category_counts = {cat: len([q for q in self.questions.values() if q.category == cat]) for cat in categories}

        return {
            'total_questions': total_questions,
            'questions_by_box': box_counts,
            'due_for_review': due_count,
            'overall_success_rate': round(overall_success_rate, 3),
            'questions_by_difficulty': difficulty_counts,
            'questions_by_category': category_counts,
            'last_updated': datetime.utcnow().isoformat()
        }

    def search_questions(self, query: str, limit: int = 10) -> List[Question]:
        """
        Search questions by content, tags, or category.

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of matching Question objects
        """
        query_lower = query.lower()
        matches = []

        for question in self.questions.values():
            # Search in content, category, tags
            if (query_lower in question.content.lower() or
                query_lower in question.category.lower() or
                any(query_lower in tag.lower() for tag in question.tags) or
                (question.answer_key and query_lower in question.answer_key.lower())):
                matches.append(question)

        # Sort by relevance (simple scoring)
        matches.sort(key=lambda q: (
            query_lower in q.content.lower(),
            query_lower in q.category.lower(),
            any(query_lower in tag.lower() for tag in q.tags)
        ), reverse=True)

        return matches[:limit]

    def add_question(self, content: str, answer_key: str = "", difficulty: int = 3,
                    category: str = "general", tags: List[str] = None) -> Question:
        """
        Manually add a question to the database.

        Args:
            content: Question text
            answer_key: Reference answer
            difficulty: Difficulty level (1-5)
            category: Question category
            tags: List of tags

        Returns:
            Created Question object
        """
        if tags is None:
            tags = []

        question_id = f"manual_{datetime.utcnow().timestamp()}_{len(self.questions):04d}"

        question = Question(
            id=question_id,
            content=content,
            answer_key=answer_key,
            difficulty=difficulty,
            category=category,
            tags=tags,
            source="manual_entry"
        )

        self.questions[question_id] = question
        self.save_database()

        logger.info(f"Added manual question: {content[:50]}...")
        return question

    def delete_question(self, question_id: str) -> bool:
        """
        Delete a question from the database.

        Args:
            question_id: ID of question to delete

        Returns:
            True if deleted successfully, False if not found
        """
        if question_id in self.questions:
            del self.questions[question_id]
            self.save_database()
            logger.info(f"Deleted question: {question_id}")
            return True

        logger.warning(f"Question not found for deletion: {question_id}")
        return False