"""
LangGraph Workflow for Interview Prep System

Implements the core interview flow with persona-based evaluation and feedback.
Uses LangGraph for state management and conditional routing.
"""

from typing import TypedDict, List, Dict, Any, Optional, Literal
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from .knowledge_engine import Question
from .persona_manager import PersonaManager
from ..rag.generator import LLMGenerator, GenerationConfig
from ..core.config import settings
from ..core.logging import logger


# ============================================================================
# State Definition
# ============================================================================

class InterviewState(TypedDict):
    """
    State for the interview workflow.

    Attributes:
        question: Current Question object from KnowledgeEngine
        user_answer: User's answer to the question
        jd_context: Job description context for generating relevant questions
        evaluation: Evaluation result from interviewer (Sukuna)
            - is_correct: bool
            - comment: str
            - score: Optional[int] (0-10 scale)
        history: Chat history for maintaining conversation context
        feedback: Feedback from tutor (Gojo) or mentor (Nanami)
        persona_mode: Which persona to use for generation/evaluation
        next_action: Next action to take in the workflow
    """
    question: Optional[Question]
    user_answer: str
    jd_context: str
    evaluation: Dict[str, Any]
    history: List[Dict[str, str]]
    feedback: str
    persona_mode: str
    next_action: Optional[str]


# ============================================================================
# LLM Helper Functions
# ============================================================================

async def _call_llm_with_persona(
    system_prompt: str,
    user_message: str,
    temperature: float = 0.7
) -> str:
    """
    Call LLM with persona-based system prompt.

    Args:
        system_prompt: System prompt (persona description)
        user_message: User message
        temperature: Generation temperature

    Returns:
        LLM response text
    """
    try:
        # Initialize text generator
        config = GenerationConfig(
            model=settings.default_llm_model,
            max_tokens=1500,
            temperature=temperature
        )

        generator = LLMGenerator(config, provider=settings.llm_provider)

        # Construct full prompt
        full_prompt = f"""{system_prompt}

User Message:
{user_message}

Response:"""

        # Generate response
        result = await generator.generate(full_prompt)

        return result.generated_text.strip()

    except Exception as e:
        logger.error(f"Failed to call LLM with persona: {e}")
        raise


# ============================================================================
# Node Functions
# ============================================================================

async def generate_node(state: InterviewState) -> InterviewState:
    """
    Generate a new question using Sukuna's persona.

    This node is called when no question exists in the state.
    Uses job description context to generate relevant questions.

    Args:
        state: Current interview state

    Returns:
        Updated state with generated question
    """
    logger.info("Executing generate_node: Generating new question")

    try:
        # Initialize persona manager
        persona_manager = PersonaManager()
        sukuna = persona_manager.get_persona("sukuna")

        if not sukuna:
            raise Exception("Sukuna persona not found")

        # Build prompt for question generation
        jd_context = state.get("jd_context", "")
        history = state.get("history", [])

        # Get conversation history context
        history_text = ""
        if history:
            recent_history = history[-3:]  # Last 3 exchanges
            history_text = "\n".join([
                f"{msg['role']}: {msg['content'][:100]}..."
                for msg in recent_history
            ])

        user_message = f"""Generate a single challenging interview question based on the following job description.

Job Description:
{jd_context[:1000]}

Previous conversation context:
{history_text if history_text else "This is the first question."}

Requirements:
1. The question should test practical knowledge and problem-solving
2. Make it specific and relevant to the job requirements
3. Include the expected answer or key points to look for
4. Rate the difficulty on a scale of 1-5

Return your response in this exact format:
QUESTION: [Your question here]
EXPECTED_ANSWER: [Key points the candidate should mention]
DIFFICULTY: [1-5]
CATEGORY: [e.g., algorithms, system_design, coding]

Now, give me a question worthy of my standards."""

        # Get system prompt from Sukuna
        system_prompt = sukuna.get_system_prompt()

        # Generate question
        response = await _call_llm_with_persona(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.8  # Higher temperature for variety
        )

        # Parse response
        question_text = ""
        expected_answer = ""
        difficulty = 3
        category = "general"

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("QUESTION:"):
                question_text = line.replace("QUESTION:", "").strip()
            elif line.startswith("EXPECTED_ANSWER:"):
                expected_answer = line.replace("EXPECTED_ANSWER:", "").strip()
            elif line.startswith("DIFFICULTY:"):
                try:
                    difficulty = int(line.replace("DIFFICULTY:", "").strip())
                except:
                    difficulty = 3
            elif line.startswith("CATEGORY:"):
                category = line.replace("CATEGORY:", "").strip()

        # Create Question object
        from datetime import datetime
        question_id = f"gen_{datetime.utcnow().timestamp()}"

        question = Question(
            id=question_id,
            content=question_text or response[:200],  # Fallback to raw response
            answer_key=expected_answer or "N/A",
            difficulty=difficulty,
            category=category,
            tags=[],
            source="sukuna_generated"
        )

        # Update state
        state["question"] = question
        state["history"].append({
            "role": "interviewer",
            "content": f"Question: {question.content}"
        })

        logger.info(f"Generated question: {question.content[:80]}...")
        return state

    except Exception as e:
        logger.error(f"generate_node failed: {e}")
        # Return state with error
        state["evaluation"] = {
            "is_correct": False,
            "comment": f"Failed to generate question: {str(e)}",
            "error": True
        }
        return state


async def evaluate_node(state: InterviewState) -> InterviewState:
    """
    Evaluate user's answer using Sukuna's persona.

    Sukuna evaluates the answer against the expected answer and provides
    harsh but fair feedback.

    Args:
        state: Current interview state

    Returns:
        Updated state with evaluation results
    """
    logger.info("Executing evaluate_node: Evaluating user answer")

    try:
        # Initialize persona manager
        persona_manager = PersonaManager()
        sukuna = persona_manager.get_persona("sukuna")

        if not sukuna:
            raise Exception("Sukuna persona not found")

        # Get question and answer
        question = state.get("question")
        user_answer = state.get("user_answer", "")

        if not question:
            raise Exception("No question to evaluate")

        if not user_answer.strip():
            # Empty answer
            state["evaluation"] = {
                "is_correct": False,
                "comment": "You dare remain silent? Pathetic. At least attempt an answer.",
                "score": 0
            }
            return state

        # Build evaluation prompt
        user_message = f"""Evaluate this candidate's answer to the interview question.

Question:
{question.content}

Expected Answer/Key Points:
{question.answer_key}

Candidate's Answer:
{user_answer}

Evaluate the answer and provide:
1. Whether it's correct/acceptable (yes/no)
2. A score from 0-10
3. Your feedback (be demanding but fair)

Return your response in this exact format:
CORRECT: [yes/no]
SCORE: [0-10]
FEEDBACK: [Your harsh but fair evaluation]

Remember your standards. Don't accept mediocrity."""

        # Get system prompt from Sukuna
        system_prompt = sukuna.get_system_prompt()

        # Generate evaluation
        response = await _call_llm_with_persona(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.3  # Lower temperature for consistent evaluation
        )

        # Parse response
        is_correct = False
        score = 0
        feedback = response  # Default to full response

        lines = response.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("CORRECT:"):
                is_correct = "yes" in line.lower()
            elif line.startswith("SCORE:"):
                try:
                    score = int(line.replace("SCORE:", "").strip())
                except:
                    score = 5
            elif line.startswith("FEEDBACK:"):
                feedback = line.replace("FEEDBACK:", "").strip()

        # Update state
        state["evaluation"] = {
            "is_correct": is_correct,
            "comment": feedback,
            "score": score
        }

        state["history"].append({
            "role": "user",
            "content": user_answer
        })

        state["history"].append({
            "role": "interviewer",
            "content": feedback
        })

        logger.info(f"Evaluation: {'correct' if is_correct else 'incorrect'}, score={score}")
        return state

    except Exception as e:
        logger.error(f"evaluate_node failed: {e}")
        state["evaluation"] = {
            "is_correct": False,
            "comment": f"Evaluation failed: {str(e)}",
            "score": 0,
            "error": True
        }
        return state


async def tutor_node(state: InterviewState) -> InterviewState:
    """
    Provide tutoring using Gojo's persona (ELI5 style).

    Called when the user's answer is incorrect. Gojo explains the concept
    in a simple, encouraging way with analogies.

    Args:
        state: Current interview state

    Returns:
        Updated state with tutor feedback
    """
    logger.info("Executing tutor_node: Providing ELI5 explanation")

    try:
        # Initialize persona manager
        persona_manager = PersonaManager()
        gojo = persona_manager.get_persona("gojo")

        if not gojo:
            raise Exception("Gojo persona not found")

        # Get question and evaluation
        question = state.get("question")
        evaluation = state.get("evaluation", {})
        user_answer = state.get("user_answer", "")

        if not question:
            raise Exception("No question to explain")

        # Build tutoring prompt
        user_message = f"""The candidate got this question wrong. Help them understand it using simple analogies and ELI5 style.

Question:
{question.content}

Expected Answer:
{question.answer_key}

Their Answer:
{user_answer}

Evaluation:
{evaluation.get('comment', 'Incorrect')}

Provide:
1. A simple explanation using analogies
2. Key concepts broken down
3. Encouragement to try again

Make it fun and easy to understand! Use your teaching superpowers!"""

        # Get system prompt from Gojo
        system_prompt = gojo.get_system_prompt()

        # Generate tutoring response
        response = await _call_llm_with_persona(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.8  # Higher temperature for creative analogies
        )

        # Update state
        state["feedback"] = response
        state["history"].append({
            "role": "tutor",
            "content": response
        })

        logger.info("Tutor feedback provided (Gojo)")
        return state

    except Exception as e:
        logger.error(f"tutor_node failed: {e}")
        state["feedback"] = f"Tutoring failed: {str(e)}"
        return state


async def mentor_node(state: InterviewState) -> InterviewState:
    """
    Provide mentoring using Nanami's persona.

    Called when the user's answer is correct. Nanami provides professional
    encouragement and guidance for continued growth.

    Args:
        state: Current interview state

    Returns:
        Updated state with mentor feedback
    """
    logger.info("Executing mentor_node: Providing professional encouragement")

    try:
        # Initialize persona manager
        persona_manager = PersonaManager()
        nanami = persona_manager.get_persona("nanami")

        if not nanami:
            raise Exception("Nanami persona not found")

        # Get question and evaluation
        question = state.get("question")
        evaluation = state.get("evaluation", {})
        user_answer = state.get("user_answer", "")
        score = evaluation.get("score", 0)

        if not question:
            raise Exception("No question to provide feedback on")

        # Build mentoring prompt
        user_message = f"""The candidate answered correctly. Provide professional encouragement and constructive feedback.

Question:
{question.content}

Their Answer:
{user_answer}

Score: {score}/10

Evaluation:
{evaluation.get('comment', 'Correct')}

Provide:
1. Recognition of their correct answer
2. What they did well
3. How to further improve or deepen understanding
4. Encouragement for continued progress

Be professional, patient, and methodical in your feedback."""

        # Get system prompt from Nanami
        system_prompt = nanami.get_system_prompt()

        # Generate mentoring response
        response = await _call_llm_with_persona(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=0.6  # Moderate temperature for balanced feedback
        )

        # Update state
        state["feedback"] = response
        state["history"].append({
            "role": "mentor",
            "content": response
        })

        logger.info("Mentor feedback provided (Nanami)")
        return state

    except Exception as e:
        logger.error(f"mentor_node failed: {e}")
        state["feedback"] = f"Mentoring failed: {str(e)}"
        return state


# ============================================================================
# Conditional Edge Functions
# ============================================================================

def should_tutor_or_mentor(state: InterviewState) -> Literal["tutor", "mentor"]:
    """
    Determine whether to route to tutor or mentor based on evaluation.

    Args:
        state: Current interview state

    Returns:
        "tutor" if answer was incorrect, "mentor" if correct
    """
    evaluation = state.get("evaluation", {})
    is_correct = evaluation.get("is_correct", False)

    if is_correct:
        logger.debug("Routing to mentor_node (answer correct)")
        return "mentor"
    else:
        logger.debug("Routing to tutor_node (answer incorrect)")
        return "tutor"


# ============================================================================
# Graph Construction
# ============================================================================

def build_interview_graph() -> CompiledStateGraph:
    """
    Build and compile the interview workflow graph.

    Workflow:
    1. Start at evaluate_node (assumes state has question and user_answer)
    2. Conditional routing based on evaluation:
       - Correct -> mentor_node -> END
       - Incorrect -> tutor_node -> END

    Returns:
        Compiled state graph ready for execution
    """
    logger.info("Building interview workflow graph")

    # Create state graph
    workflow = StateGraph(InterviewState)

    # Add nodes
    workflow.add_node("generate", generate_node)
    workflow.add_node("evaluate", evaluate_node)
    workflow.add_node("tutor", tutor_node)
    workflow.add_node("mentor", mentor_node)

    # Set entry point
    workflow.set_entry_point("evaluate")

    # Add conditional edge based on evaluation result
    workflow.add_conditional_edges(
        "evaluate",
        should_tutor_or_mentor,
        {
            "tutor": "tutor",
            "mentor": "mentor"
        }
    )

    # Add edges to END
    workflow.add_edge("tutor", END)
    workflow.add_edge("mentor", END)

    # Note: generate node is available but not part of this flow
    # It can be called separately when needed

    # Compile graph
    app = workflow.compile()

    logger.info("Interview workflow graph compiled successfully")
    return app


# ============================================================================
# Export
# ============================================================================

# Create and export the compiled graph
app = build_interview_graph()

__all__ = [
    "InterviewState",
    "app",
    "build_interview_graph",
    "generate_node",
    "evaluate_node",
    "tutor_node",
    "mentor_node",
]
