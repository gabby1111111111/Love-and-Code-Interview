"""
Generator for Interview Prep System

Implements "Polyphonic Questioning" and "Three-Fold Judgment" logic.
Orchestrates the interaction between the Persona, the Question, and the LLM.
"""

import json
import re
from typing import Dict, Any, Optional, List

from ..core.logging import logger
from ..core.config import settings
from ..rag.generator import LLMGenerator, GenerationConfig
from .persona_manager import Persona
from .knowledge_engine import Question


class Generator:
    """
    Manages LLM generation for the interview flow.

    Features:
    - Polyphonic Questioning: Generates questions with role-play flavor + technical clarity.
    - Three-Fold Judgment: Generates feedback with Character Verdict + Standard Answer + ELI5.
    - JSON Output Enforcement: Ensures LLM outputs structured data for the UI.
    """

    def __init__(self):
        self.config = GenerationConfig(
            model=settings.default_llm_model,
            max_tokens=2000,
            temperature=0.7
        )
        self.llm = LLMGenerator(self.config, provider=settings.llm_provider)

    async def generate_question_interaction(
        self,
        persona: Persona,
        question: Question,
        jd_context: str = "",
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate the "Polyphonic Question" object.

        Args:
            persona: The character persona
            question: The technical question to ask
            jd_context: Optional job description context
            language: Output language code (e.g., "en", "zh")

        Returns:
            JSON object with lore_flavor, original_question, and hints.
        """
        system_prompt = persona.get_system_prompt()

        lang_instruction = "English" if language == "en" else "Chinese (Simplified)"
        
        # Extract world lore for ELI5
        world_lore = ""
        if persona.character_book and isinstance(persona.character_book, dict) and 'entries' in persona.character_book:
            try:
                # Get first 3 lore entries (handle both list and dict formats)
                entries_data = persona.character_book['entries']
                
                if hasattr(entries_data, 'values'):
                    entries = list(entries_data.values())[:3]
                elif isinstance(entries_data, list):
                    entries = entries_data[:3]
                else:
                    entries = []
                    
                world_lore = " ".join([e.get('content', '') for e in entries if isinstance(e, dict) and 'content' in e])[:500]
            except Exception as e:
                print(f"Error processing character_book: {e}")
                world_lore = ""

        user_message = f"""YOU ARE THE SCENARIO ENGINE. You are NOT the character.

**Critical Directive:** The character ({persona.name}) is the INTERVIEWER. The USER is the INTERVIEWEE (candidate).
You must generate text that {persona.name} would SAY TO the user, not what the user would say.

[Character Context]
- Name: {persona.name}
- Role: {persona.role}
- World: {persona.scenario or "Generic interview setting"}
- World Lore Summary: {world_lore or "No specific lore"}

[Technical Question]
{question.content}

[Job Context]
{jd_context[:500]}

[Your Task]
Generate a JSON object representing how {persona.name} would present this question in their world's context.

**DIALOGUE FORMAT RULES:**
All dialogue MUST follow this format:
"{persona.name}：（动作/心理描写）\\"对话内容\\""

Example formats:
- "大E：（冷峻地注视着你）\\"证明你的价值。\\""
- "五条悟：（轻笑着）\\"这题对你来说太简单了吧？\\""
- "宿傩：（不屑地翘起嘴角）\\"蝼蚁，别让我失望。\\""

**JSON Structure:**
{{
  "lore_flavor": "{persona.name}：（动作/心理描写）\\"A majestic command relating the technical topic to their world. MUST start with character name and action.\\"",
  "original_question": "{question.content}",
  "tech_hint": "3-5 technical keywords separated by commas.",
  "eli5_hint": "A simplified analogy using {persona.name}'s WORLD LORE. MUST use concepts from their world. Example: 'Machine Spirit protocols' instead of 'computer programs'.",
  "encouragement": "{persona.name}：（简短动作）\\"一句话\\""
}}

[CRITICAL Requirements]
1. lore_flavor and encouragement MUST include character name prefix
2. MUST include action/emotion in （）
3. eli5_hint MUST use world-specific terminology from the lore
4. Generic Earth analogies are FORBIDDEN

[Language Requirement]
All generated text must be in {lang_instruction}.

[Constraint]
Do NOT answer the question. Just present it as a challenge from {persona.name} to the user.
Output ONLY the JSON object, no markdown blocks.
"""

        try:
            result = await self.llm.generate(f"{system_prompt}\n\nUser: {user_message}")
            return self._parse_json_response(result.generated_text)
        except Exception as e:
            logger.error(f"Error generating question interaction: {e}")
            # Fallback
            fallback_lore = f"{persona.name} 审视着你。" if language == "zh" else f"{persona.name} gazes at you."
            return {
                "lore_flavor": fallback_lore,
                "original_question": question.content,
                "tech_hint": "N/A",
                "eli5_hint": "暂无类比。" if language == "zh" else "No analogy available.",
                "encouragement": "回答。" if language == "zh" else "Answer."
            }

    async def generate_dual_question_interaction(
        self,
        emperor_persona: Persona,
        tutor_persona: Persona,
        question: Question,
        jd_context: str = "",
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate concurrent question interaction from both Emperor and a Tutor."""
        lang_instruction = "English" if language == "en" else "Chinese (Simplified)"
        
        # 1. Emperor Prompt (Strict Questioning)
        emp_sys = emperor_persona.get_system_prompt()
        emp_user = f"""YOU ARE THE SCENARIO ENGINE. The character ({emperor_persona.name}) is the SUPREME EXAMINER.
[Technical Question]
{question.content}

Generate a JSON object representing how {emperor_persona.name} coldly asks this question.
**Format**: "{emperor_persona.name}：（动作/心理描写）\\"对话内容\\""
{{
  "lore_flavor": "{emperor_persona.name}：（动作/心理描写）\\"A majestic command to answer the question.\\"",
  "original_question": "{question.content}"
}}
Must be in {lang_instruction}. Output ONLY JSON.
"""

        # 2. Tutor Prompt (Teaching & Hints)
        tut_sys = tutor_persona.get_system_prompt()
        tut_user = f"""YOU ARE THE SCENARIO ENGINE. The character ({tutor_persona.name}) is the TUTOR whispering advice to the candidate.
[Technical Question]
{question.content}

[Your Task]
Generate a JSON object representing how {tutor_persona.name} react to the Emperor's question and secretly gives the candidate a hint.
**Format**: "{tutor_persona.name}：（动作/心理描写）\\"对话内容\\""
{{
  "lore_flavor": "{tutor_persona.name}：（动作/心理描写）\\"In-character reaction to the question and encouraging the user.\\"",
  "tech_hint": "3-5 technical keywords separated by commas.",
  "eli5_hint": "A simplified analogy in {tutor_persona.name}'s voice."
}}
Must be in {lang_instruction}. Output ONLY JSON.
"""

        import asyncio
        import json
        
        async def fetch_emp():
            try:
                res = await self.llm.generate(f"{emp_sys}\n\nUser: {emp_user}")
                return self._parse_json_response(res.generated_text)
            except Exception:
                fallback_lore = emperor_persona.name + "：（凝视着你）\"回答这个问题。\""
                return {"lore_flavor": fallback_lore, "original_question": question.content}

        async def fetch_tut():
            try:
                res = await self.llm.generate(f"{tut_sys}\n\nUser: {tut_user}")
                return self._parse_json_response(res.generated_text)
            except Exception:
                fallback_lore = tutor_persona.name + "：（拍拍你的肩膀）\"别紧张，发挥你的实力。\""
                return {"lore_flavor": fallback_lore, "tech_hint": "N/A", "eli5_hint": "N/A"}

        emp_res, tut_res = await asyncio.gather(fetch_emp(), fetch_tut())
        
        # Combine results
        return {
            "emperor_flavor": emp_res.get("lore_flavor", ""),
            "original_question": emp_res.get("original_question", question.content),
            "tutor_flavor": tut_res.get("lore_flavor", ""),
            "tech_hint": tut_res.get("tech_hint", ""),
            "eli5_hint": tut_res.get("eli5_hint", "")
        }

    async def generate_story_node(
        self,
        persona: Persona,
        node_type: str,
        success_rate: float,
        language: str = "zh"
    ) -> Dict[str, Any]:
        """
        Generate story node content for dramatic moments.

        Args:
            persona: Character persona
            node_type: "node_a" or "node_b"
            success_rate: Player's success rate (0.0-1.0)
            language: Output language

        Returns:
            JSON with story content
        """
        system_prompt = persona.get_system_prompt()
        lang_instruction = "Chinese (Simplified)" if language == "zh" else "English"

        if node_type == "node_a":
            prompt = f"""You are the cinematic director for {persona.name}'s story.

The candidate has completed 2 technical questions. Now trigger STORY NODE A: Gene Surgery / First Warp Contact.

Generate a dramatic, immersive story segment (200-300 words) in {lang_instruction} where:
1. {persona.name} pauses the interview
2. Describes a transformative ritual/test (gene surgery, warp exposure, etc.)
3. Builds tension and world lore
4. Ends with: "Prove yourself worthy. Continue."

Output only the story text, no JSON."""

        else:  # node_b
            if success_rate >= 0.8:
                outcome = "ascension"
                prompt_detail = "triumphantly inducted as a full Astartes/honored warrior"
            elif success_rate >= 0.5:
                outcome = "survival"
                prompt_detail = "barely survives, scarred but accepted as a scout"
            else:
                outcome = "chaos"
                prompt_detail = "fails, consumed by Chaos/rejected and mutated"

            prompt = f"""You are the cinematic director for {persona.name}'s final judgment.

The candidate's success rate: {success_rate:.1%}
Outcome: {outcome}

Generate a dramatic ENDING (300-400 words) in {lang_instruction} where:
1. {persona.name} delivers final judgment
2. Describes the candidate's fate: {prompt_detail}
3. Epic, cinematic language
4. Conclude their story arc

Output only the story text, no JSON."""

        try:
            result = await self.llm.generate(prompt)
            return {"story_content": result.generated_text}
        except Exception as e:
            logger.error(f"Error generating story node: {e}")
            return {"story_content": f"{persona.name} 凝视着你，沉默不语。" if language == "zh" else f"{persona.name} gazes at you in silence."}

    async def generate_feedback(
        self,
        persona: Persona,
        question: Question,
        user_answer: str,
        evaluation: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """
        Generate the "Three-Fold Judgment" feedback.

        Args:
            persona: The character persona
            question: The technical question
            user_answer: The user's answer
            evaluation: Previous evaluation result (optional)
            language: Output language code

        Returns:
            JSON object with verdict, standard_answer, servitor_explanation.
        """
        system_prompt = persona.get_system_prompt()

        lang_instruction = "English" if language == "en" else "Chinese (Simplified)"

        user_message = f"""YOU ARE THE JUDGMENT ENGINE. You are NOT the character.

**Critical Directive:**
- The character ({persona.name}) is the SUPREME JUDGE.
- The USER is the candidate who just answered.
- Generate a verdict from {persona.name} SPEAKING TO/ABOUT the user, NOT from the user's perspective.

[Technical Question]
{question.content}

[Expected Answer]
{question.answer_key}

[Candidate's Answer]
{user_answer}

[Your Task]
Generate a JSON object with THREE distinct judgments:

1. **verdict**: {persona.name}'s direct reaction to the user's answer.
   - MUST follow format: "{persona.name}：（动作/心理）\\"评价内容\\""
   - If WRONG: Scold the USER for failing
   - If RIGHT: Grant the USER honor/approval  
   - If PARTIAL: Express disappointment but acknowledge effort
   - CRITICAL: Do NOT criticize {persona.name}. You are {persona.name} judging the USER.

2. **standard_answer**: The boring, correct technical answer (out of character).

3. **servitor_explanation**: A simple "Explain Like I'm 5" version for teaching.

**JSON Structure:**
{{
  "verdict": {{
    "status": "correct" | "incorrect" | "partial",
    "comment": "{persona.name}：（vivid action/emotion）\\"in-character judgment of the USER\\""
  }},
  "standard_answer": "Clean technical answer.",
  "servitor_explanation": "Simple child-friendly explanation."
}}

[Example verdict.comment formats]
- "{persona.name}：（满意地点头）\\"不错，你通过了。\\""
- "{persona.name}：（冷笑）\\"愚蠢。你让我失望了。\\""
- "{persona.name}：（沉思片刻）\\"勉强及格，继续努力。\\""

[Language Requirement]
All text must be in {lang_instruction}.

Output ONLY the JSON object, no markdown blocks.
"""

        try:
            result = await self.llm.generate(f"{system_prompt}\n\nUser: {user_message}")
            return self._parse_json_response(result.generated_text)
        except Exception as e:
            logger.error(f"Error generating feedback: {e}")
            return {
                "verdict": {
                    "status": "partial",
                    "comment": "无法评估。" if language == "zh" else "Cannot evaluate."
                },
                "standard_answer": question.answer_key or "N/A",
                "servitor_explanation": "暂无解释。" if language == "zh" else "N/A"
            }

    async def generate_dual_feedback(
        self,
        emperor_persona: Persona,
        tutor_persona: Persona,
        question: Question,
        user_answer: str,
        evaluation: Dict[str, Any],
        language: str = "en"
    ) -> Dict[str, Any]:
        """Generate concurrent feedback from Emperor (Verdict) and Tutor (Explanation)."""
        lang_instruction = "English" if language == "en" else "Chinese (Simplified)"

        # Context Extraction
        # Fall back to answer_key if specific context doesn't exist
        pro_context = getattr(question, 'pro_context', None) or question.answer_key
        gabriella_context = getattr(question, 'gabriella_context', None) or question.answer_key

        # 1. Emperor Prompt (Verdict Only)
        emp_sys = emperor_persona.get_system_prompt()
        emp_user = f"""YOU ARE THE JUDGMENT ENGINE. Character: {emperor_persona.name}.
[Technical Question]
{question.content}
[Expected Professional Answer]
{pro_context}
[Candidate's Answer]
{user_answer}

Generate a JSON object with {emperor_persona.name}'s verdict.
{{
  "verdict": {{
    "status": "correct" | "incorrect" | "partial",
    "comment": "{emperor_persona.name}：（动作/心理）\\"Critique the Candidate's answer strictly based on the Expected Professional Answer.\\""
  }}
}}
Must be in {lang_instruction}. Output ONLY JSON.
"""

        # 2. Tutor Prompt (Standard Answer and ELI5 Explanation)
        tut_sys = tutor_persona.get_system_prompt()
        tut_user = f"""YOU ARE THE JUDGMENT ENGINE. Character: {tutor_persona.name}.
[Technical Question]
{question.content}
[Cyber Tea Party Conversational Context]
{gabriella_context}
[Professional Answer]
{pro_context}
[Candidate's Answer]
{user_answer}

Generate a JSON object with {tutor_persona.name}'s friendly explanation and standard answer.
{{
  "standard_answer": "Extract the core technical points from the Professional Answer.",
  "servitor_explanation": "{tutor_persona.name}：（动作/心理）\\"Translate the ideas from the 'Cyber Tea Party Conversational Context' into your own character's unique speaking style to help the candidate understand.\\""
}}
Must be in {lang_instruction}. Output ONLY JSON.
"""

        import asyncio
        
        async def fetch_emp():
            try:
                res = await self.llm.generate(f"{emp_sys}\n\nUser: {emp_user}")
                return self._parse_json_response(res.generated_text)
            except Exception:
                fallback_comment = emperor_persona.name + "：（沉默不语）"
                return {"verdict": {"status": "partial", "comment": fallback_comment}}

        async def fetch_tut():
            try:
                res = await self.llm.generate(f"{tut_sys}\n\nUser: {tut_user}")
                return self._parse_json_response(res.generated_text)
            except Exception:
                return {"standard_answer": question.answer_key, "servitor_explanation": "网络断开中..."}

        emp_res, tut_res = await asyncio.gather(fetch_emp(), fetch_tut())
        
        # Combine
        return {
            "verdict": emp_res.get("verdict", {"status": "partial", "comment": ""}),
            "standard_answer": tut_res.get("standard_answer", question.answer_key),
            "servitor_explanation": tut_res.get("servitor_explanation", "")
        }

    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        try:
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            text = text.strip()
            return json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON from LLM: {text[:100]}...")
            # Try to find JSON-like structure
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except:
                    pass
            raise ValueError("Could not parse JSON response")
