"""
Gamified Interview Prep System

A comprehensive interview preparation system with:
- SillyTavern Character Card support for personas
- Spaced repetition learning algorithm
- LLM-powered question generation
- LangGraph workflow for interactive interviews
- Progress tracking and analytics
"""

from .knowledge_engine import KnowledgeEngine, Question
from .persona_manager import PersonaManager, Persona
from .generator import Generator
try:
    from .graph import (
        InterviewState,
        app,
        build_interview_graph,
        generate_node,
        evaluate_node,
        tutor_node,
        mentor_node,
    )
except ImportError:
    # LangGraph not installed or graph.py has errors
    # Define dummies or just pass if not used
    InterviewState = None
    app = None
    build_interview_graph = None
    generate_node = None
    evaluate_node = None
    tutor_node = None
    mentor_node = None

__all__ = [
    # Knowledge Engine
    "KnowledgeEngine",
    "Question",
    # Persona Management
    "PersonaManager",
    "Persona",
    # LangGraph Workflow
    "InterviewState",
    "app",
    "build_interview_graph",
    "generate_node",
    "evaluate_node",
    "tutor_node",
    "mentor_node",
    # Generator
    "Generator",
]
