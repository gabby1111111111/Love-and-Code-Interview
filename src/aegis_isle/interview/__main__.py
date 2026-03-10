import asyncio
from pathlib import Path

from .persona_manager import PersonaManager
from .knowledge_engine import KnowledgeEngine
from .generator import Generator

async def main():
    print("Initializing Interview Prep System...")
    
    # 1. Initialize Managers
    data_dir = Path("e:/Love-and-Code-Interview/data")
    persona_manager = PersonaManager(persona_dir=data_dir / "personas")
    knowledge_engine = KnowledgeEngine(db_path=data_dir / "interview_db.json")
    generator = Generator()
    
    # 2. Get Persona
    persona_name = "gojo" # default testing persona
    persona = persona_manager.get_persona(persona_name)
    if not persona:
        persona = persona_manager.get_default_persona()
    print(f"\n[{persona.role}] {persona.name} has joined the session.")
    print("-" * 50)
    print(persona.first_message)
    print("-" * 50)
    
    # 3. Get Question
    question = knowledge_engine.get_next_question()
    if not question:
        print("\nNo questions available. Adding a sample question...")
        question = knowledge_engine.add_question(
            content="什么是多态？在Python中如何实现？",
            answer_key="多态是指不同类型的对象对同一消息作出不同响应的能力。Python通过鸭子类型（duck typing）隐式实现多态，只要对象实现了期望的方法即可被调用。",
            difficulty=2,
            category="python"
        )
    
    # 4. Generate Interaction Context
    print("\nGenerating character question context...")
    try:
        interaction = await generator.generate_question_interaction(
            persona=persona, 
            question=question,
            language="zh"
        )
        print("\n\033[96m" + interaction.get("lore_flavor", question.content) + "\033[0m")
        print(f"\n\033[93m[Hint: {interaction.get('tech_hint', 'N/A')}]\033[0m")
    except Exception as e:
        print(f"Error generating interaction: {e}")
        print("\nQuestion:", question.content)
    
    # 5. User Input
    print("\nYour answer (press enter to skip):")
    user_answer = input("> ")
    if not user_answer.strip():
        user_answer = "我不确定，请告诉我标准答案。"
        print(f"Skipping... Using fallback answer: {user_answer}")
    
    # 6. Evaluation and Feedback
    print("\nEvaluating answer...")
    try:
        feedback = await generator.generate_feedback(
            persona=persona,
            question=question,
            user_answer=user_answer,
            evaluation={},
            language="zh"
        )
        
        verdict = feedback.get("verdict", {})
        print("\n\033[91mVerdict:\033[0m")
        print(verdict.get("comment", "No comment."))
        
        print("\n\033[92mStandard Answer:\033[0m")
        print(feedback.get("standard_answer", question.answer_key))
        
        print("\n\033[94mExplanation (ELI5):\033[0m")
        print(feedback.get("servitor_explanation", "No explanation available."))
        
        # Update progress based on the result
        is_correct = verdict.get("status") == "correct"
        knowledge_engine.update_progress(question.id, is_correct)
        
    except Exception as e:
        print(f"Error generating feedback: {e}")

if __name__ == "__main__":
    import os
    if "PYTHONPATH" not in os.environ:
        import sys
        # ensure local modules are discoverable
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    asyncio.run(main())
