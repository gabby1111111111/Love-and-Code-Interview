"""
Persona Manager for Interview Prep System

Supports loading SillyTavern Character Cards (V2 Spec) from JSON or PNG files.
Provides default personas for different interview roles.
"""

import json
import base64
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from PIL import Image

from ..core.logging import logger


@dataclass
class Persona:
    """
    Persona data model representing an interview character.

    Attributes:
        name: Character name
        role: Interview role (Interviewer, Tutor, Mentor)
        description: Character background and description
        personality: Personality traits and behavior patterns
        first_message: Initial greeting message (Cinematic Intro)
        example_messages: Example conversation snippets
        scenario: The current situation/scene
        character_book: World info / lore entries
        avatar_path: Optional path to avatar image
    """
    name: str
    role: str
    description: str
    personality: str
    first_message: str
    example_messages: str
    scenario: str = ""
    character_book: Optional[Dict[str, Any]] = None
    avatar_path: Optional[str] = None

    def get_system_prompt(self) -> str:
        """
        Generate a powerful system prompt for the LLM instructing it to act
        as an Interviewer within the specific World.
        """
        # Format World Info if available
        world_info_text = ""
        if self.character_book and 'entries' in self.character_book:
            entries = self.character_book['entries']
            if entries:
                world_info_text = "World Info / Lore:\n"
                # Fix: iterate over values if entries is a dict, otherwise iterate directly
                entries_list = entries.values() if isinstance(entries, dict) else entries
                for entry in entries_list:
                    if isinstance(entry, dict):  # Safety check
                        keys = ", ".join(entry.get('keys', []))
                        content = entry.get('content', '')
                        if content:
                            world_info_text += f"- [{keys}]: {content}\n"
                world_info_text += "\n"

        return f"""You are {self.name}, acting as a {self.role}.

[Character Description]
{self.description}

[Personality]
{self.personality}

[Scenario / Current Situation]
{self.scenario}

{world_info_text}[Roleplay Instructions]
Your role is to conduct a technical interview, but you MUST stay completely in character.
- Adopt the tone, mannerisms, and worldview of {self.name}.
- If the world is fantasy/sci-fi, frame technical concepts using analogies from that world if appropriate, or treat the interview as a test of skill/magic/competence within that setting.
- Do NOT break character.
- Be {self.role} first, and a technical evaluator second.

[Example Dialogue]
{self.example_messages}
"""


class PersonaManager:
    """
    Manages interview personas and SillyTavern Character Card loading.

    Supports:
    - Loading character cards from JSON files
    - Extracting character data from PNG metadata (SillyTavern V2 format)
    - Default hardcoded personas (Sukuna, Gojo, Nanami)
    """

    def __init__(self, persona_dir: Optional[Path] = None):
        """
        Initialize PersonaManager.

        Args:
            persona_dir: Directory containing character card files (optional)
        """
        self.persona_dir = persona_dir or Path("data/personas")
        self.personas: Dict[str, Persona] = {}
        self._initialize_default_personas()

        # Load custom personas if directory exists
        if self.persona_dir.exists():
            self._load_custom_personas()

        logger.info(f"PersonaManager initialized with {len(self.personas)} personas")

    def _initialize_default_personas(self):
        """Initialize hardcoded default personas."""

        # Sukuna - Strict Interviewer
        sukuna = Persona(
            name="两面宿傩",
            role="Interviewer",
            description="The King of Curses, known for his ruthless efficiency and high standards. "
                       "In this context, Sukuna is a demanding technical interviewer who doesn't "
                       "accept mediocrity. He pushes candidates to their limits.",
            personality="Direct, intimidating, uncompromising, expects excellence. "
                       "Doesn't tolerate weak answers. Respects strength and competence. "
                       "May mock poor answers but acknowledges good ones.",
            first_message="So, you dare to face me in an interview? Very well. "
                         "Let's see if you have what it takes. Don't waste my time with mediocre answers.",
            example_messages="""User: "I think the time complexity is O(n)..."
Sukuna: "You THINK? That's not good enough. Either you KNOW or you don't. Weak."

User: "The optimal solution uses dynamic programming with memoization."
Sukuna: "Hmph. Finally, a competent answer. Continue."

User: "I'm not sure about this one..."
Sukuna: "Uncertainty is weakness. Think harder or admit defeat."
""",
            scenario="Sukuna sits on his throne of skulls, looking down at the candidate."
        )

        # Gojo - Playful Tutor
        gojo = Persona(
            name="五条悟",
            role="Tutor",
            description="The strongest sorcerer, known for his playful demeanor and exceptional teaching ability. "
                       "Gojo makes complex concepts easy to understand with his ELI5 (Explain Like I'm 5) approach. "
                       "He's encouraging but won't let you slack off.",
            personality="Playful, confident, encouraging, uses analogies and simple explanations. "
                       "Makes learning fun. Sometimes teases but always supportive. "
                       "Breaks down complex topics into digestible pieces.",
            first_message="Yo! Ready to learn something cool? Don't worry, I'll make this super easy to understand. "
                         "No question is too simple for the strongest teacher around! 😎",
            example_messages="""User: "I don't understand recursion..."
Gojo: "Ah, recursion! Think of it like looking in a mirror that reflects another mirror. Each reflection is a function calling itself! Let me break it down step by step."

User: "What's the difference between a stack and queue?"
Gojo: "Easy! A stack is like a stack of pancakes - you eat from the top (LIFO). A queue is like a line at a coffee shop - first person in line gets served first (FIFO). See? Simple! ☕"

User: "This is hard..."
Gojo: "Everything seems hard until you understand it! Let's tackle this together. You've got this!"
""",
            scenario="Gojo is lounging in a classroom chair, spinning a pen."
        )

        # Nanami - Encouraging Mentor
        nanami = Persona(
            name="七海建人",
            role="Mentor",
            description="A professional and methodical sorcerer who values work-life balance and proper technique. "
                       "Nanami is a patient mentor who focuses on building strong foundations and sustainable growth. "
                       "He encourages structured learning and realistic goal-setting.",
            personality="Professional, patient, methodical, realistic, encouraging. "
                       "Values proper fundamentals. Provides constructive feedback. "
                       "Believes in steady progress over rushing. Respects effort.",
            first_message="Good to see you're taking your interview preparation seriously. "
                         "Let's work through this methodically. I'll help you build a strong foundation, "
                         "one step at a time. Remember, consistent effort leads to success.",
            example_messages="""User: "I got the answer wrong..."
Nanami: "That's part of the learning process. Let's analyze where the thinking went wrong and correct it. Mistakes are valuable learning opportunities."

User: "Should I learn everything at once?"
Nanami: "Absolutely not. That's a recipe for burnout. Focus on mastering fundamentals first, then gradually expand your knowledge. Quality over quantity."

User: "I'm ready for the next challenge!"
Nanami: "Good attitude. But let's make sure you've truly mastered this concept first. A strong foundation is crucial for long-term success."
""",
            scenario="Nanami is reviewing documents at his desk, looking up to address the candidate."
        )

        # Add default personas to registry
        self.personas["sukuna"] = sukuna
        self.personas["gojo"] = gojo
        self.personas["nanami"] = nanami

        logger.info("Default personas initialized: Sukuna, Gojo, Nanami")

    def _load_custom_personas(self):
        """Load custom personas from persona directory."""
        try:
            for file_path in self.persona_dir.glob("*"):
                if file_path.suffix.lower() in [".json", ".png"]:
                    try:
                        persona = self.load_card(file_path)
                        persona_key = persona.name.lower().replace(" ", "_")
                        self.personas[persona_key] = persona
                        logger.info(f"Loaded custom persona: {persona.name}")
                    except Exception as e:
                        logger.warning(f"Failed to load persona from {file_path}: {e}")
        except Exception as e:
            logger.warning(f"Error loading custom personas: {e}")

    def load_card(self, file_path: Path) -> Persona:
        """
        Load a SillyTavern Character Card from JSON or PNG file.

        Supports:
        - Direct JSON character card files
        - PNG files with embedded character data in metadata

        Args:
            file_path: Path to character card file (JSON or PNG)

        Returns:
            Persona object with extracted character data

        Raises:
            ValueError: If file format is unsupported or data is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not file_path.exists():
            raise FileNotFoundError(f"Character card file not found: {file_path}")

        file_extension = file_path.suffix.lower()

        if file_extension == ".json":
            return self._load_from_json(file_path)
        elif file_extension == ".png":
            return self._load_from_png(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}. "
                           "Only .json and .png files are supported.")

    def _load_from_json(self, file_path: Path) -> Persona:
        """Load character data from JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            return self._parse_character_data(data, str(file_path))
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in character card: {e}")
        except Exception as e:
            raise ValueError(f"Error loading JSON character card: {e}")

    def _load_from_png(self, file_path: Path) -> Persona:
        """
        Load character data from PNG metadata.

        Checks for 'ccv3' (Character Card V3) or 'chara' metadata fields
        which contain base64-encoded JSON character data.
        """
        try:
            with Image.open(file_path) as img:
                # Check for metadata in PNG info
                metadata = img.info

                # Try 'ccv3' field first (newer format)
                if 'ccv3' in metadata:
                    char_data_str = metadata['ccv3']
                    logger.debug(f"Found 'ccv3' metadata in {file_path}")
                # Fall back to 'chara' field (older format)
                elif 'chara' in metadata:
                    char_data_str = metadata['chara']
                    logger.debug(f"Found 'chara' metadata in {file_path}")
                else:
                    raise ValueError(f"No character card metadata found in PNG. "
                                   f"Expected 'ccv3' or 'chara' field.")

                # Decode base64 data
                try:
                    char_data_json = base64.b64decode(char_data_str).decode('utf-8')
                    data = json.loads(char_data_json)
                except Exception as e:
                    raise ValueError(f"Failed to decode character data: {e}")

                return self._parse_character_data(data, str(file_path))

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Error loading PNG character card: {e}")

    def _parse_character_data(self, data: Dict[str, Any], source_path: str) -> Persona:
        """
        Parse character data from SillyTavern V2 spec.

        Expected fields:
        - name: Character name (required)
        - description: Character background (required)
        - personality: Personality traits (optional)
        - first_mes: First message (optional)
        - mes_example: Example messages (optional)
        - scenario: Scenario (optional)
        - character_book: World info (optional)
        - data: Alternative location for character data (nested format)
        """
        # Handle nested 'data' format
        if 'data' in data and isinstance(data['data'], dict):
            data = data['data']

        # Extract required fields
        name = data.get('name', '').strip()
        if not name:
            raise ValueError("角色卡缺少必需的 'name' 字段 (Character card missing required 'name' field)")

        # Extract optional fields with defaults
        # Description is now optional - fallback to personality or a generic default
        description = data.get('description', data.get('desc', '')).strip()
        personality = data.get('personality', '').strip()
        
        # If no description, use personality, or a generic fallback
        if not description:
            if personality:
                description = f"性格: {personality}" if personality else "A mysterious character."
            else:
                description = "一个神秘的角色。(A mysterious character.)"
                logger.warning(f"Character '{name}' has no description or personality. Using default.")

        first_message = data.get('first_mes', data.get('greeting', '')).strip()
        example_messages = data.get('mes_example', data.get('example_dialogue', '')).strip()
        scenario = data.get('scenario', '').strip()
        character_book = data.get('character_book', None)

        # Determine role based on personality or description
        role = self._infer_role(name, description, personality)

        # Create persona
        persona = Persona(
            name=name,
            role=role,
            description=description,
            personality=personality or "Professional and helpful",
            first_message=first_message or f"Hello! I'm {name}. Let's prepare for your interview!",
            example_messages=example_messages or "No example dialogue provided.",
            scenario=scenario,
            character_book=character_book,
            avatar_path=source_path if source_path.endswith('.png') else None
        )

        logger.info(f"Successfully parsed character card: {name} ({role})")
        return persona

    def _infer_role(self, name: str, description: str, personality: str) -> str:
        """
        Infer the role (Interviewer/Tutor/Mentor) based on character attributes.

        Uses keyword matching in name, description, and personality.
        """
        combined_text = f"{name} {description} {personality}".lower()

        # Keywords for each role
        interviewer_keywords = ['interview', 'strict', 'demanding', 'ruthless', 'challenging']
        tutor_keywords = ['teach', 'tutor', 'explain', 'playful', 'simple', 'eli5']
        mentor_keywords = ['mentor', 'guide', 'patient', 'encourage', 'support', 'professional']

        # Count keyword matches
        interviewer_score = sum(1 for kw in interviewer_keywords if kw in combined_text)
        tutor_score = sum(1 for kw in tutor_keywords if kw in combined_text)
        mentor_score = sum(1 for kw in mentor_keywords if kw in combined_text)

        # Determine role based on highest score
        max_score = max(interviewer_score, tutor_score, mentor_score)

        if max_score == 0:
            return "Mentor"  # Default role
        elif interviewer_score == max_score:
            return "Interviewer"
        elif tutor_score == max_score:
            return "Tutor"
        else:
            return "Mentor"

    def get_persona(self, name: str) -> Optional[Persona]:
        """
        Get a persona by name.

        Args:
            name: Persona name (case-insensitive)

        Returns:
            Persona object if found, None otherwise
        """
        persona_key = name.lower().replace(" ", "_")
        persona = self.personas.get(persona_key)

        if persona:
            logger.debug(f"Retrieved persona: {persona.name}")
        else:
            logger.warning(f"Persona not found: {name}")

        return persona

    def list_personas(self) -> list[str]:
        """
        Get list of all available persona names.

        Returns:
            List of persona names
        """
        return [persona.name for persona in self.personas.values()]

    def get_default_persona(self) -> Persona:
        """
        Get the default persona (Gojo - balanced and friendly).

        Returns:
            Default Persona object
        """
        return self.personas["gojo"]
