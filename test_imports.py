import sys
import os

sys.path.insert(0, r'e:\Love-and-Code-Interview\src')
try:
    from aegis_isle.interview.generator import Generator
    from aegis_isle.interview.knowledge_engine import KnowledgeEngine
    from aegis_isle.interview.persona_manager import PersonaManager
    print('Imports success!')
except Exception as e:
    import traceback
    traceback.print_exc()
