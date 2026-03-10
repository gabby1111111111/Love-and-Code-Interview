"""
Interview Prep System - The Infinite Interview
"Project Love & Code"

A Cinematic, Otome-Game Style, Infinite Role-Play Interview System.
"""

import sys
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import base64
import os

import streamlit as st

# Add src to path for imports - get absolute path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
src_path = project_root / "src"

# Add to Python path if not already there
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

# Also add project root
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from aegis_isle.interview import (
    KnowledgeEngine,
    PersonaManager,
    Question,
    Generator
)
from aegis_isle.interview.story_manager import StoryManager


# ============================================================================
# Language Configurationren
# ============================================================================

TRANSLATIONS = {
    "en": {
        "title": "✨ The Infinite Interview ✨",
        "subtitle": "A Cinematic Role-Play Experience",
        "sidebar_config": "⚙️ Configuration",
        "lang_select": "🌐 Language",
        "jd_section": "📄 Job Description",
        "jd_label": "Enter Job Description",
        "jd_help": "Paste the job description to generate relevant questions",
        "kb_section": "📚 Knowledge Base",
        "kb_upload": "Upload Study Material",
        "kb_help": "Upload text file to generate interview questions",
        "kb_process": "📥 Process Knowledge Base",
        "kb_success": "Generated {} questions from knowledge base!",
        "card_section": "🎭 Character Card",
        "card_upload": "Upload Character Card",
        "card_help": "Upload SillyTavern character card (JSON or PNG)",
        "card_load": "📥 Summon Character",
        "card_success": "Summoned: {} ({})",
        "card_error": "Error summoning character: {}",
        "start_button": "🎬 Enter the World",
        "submit_answer": "📤 Submit Response",
        "next_question": "⏭️ Next Challenge",
        "feedback_title": "Judgment",
        "correct": "✅ Correct",
        "incorrect": "❌ Incorrect",
        "partial": "⚠️ Partial",
        "score": "Score",
        "loading": "The world is shifting...",
        "no_questions": "The void is empty. (Add questions via Knowledge Base)",
        "intro_placeholder": "Type your role-play response...",
        "answer_placeholder": "Speak your answer...",
        "retry": "Retry",
        "hints_title": "💡 Hints & Analogies",
        "keywords": "Keywords:",
        "eli5": "ELI5:",
        "tech_q": "Technical Question:",
        "std_ans": "Standard Answer:",
        "config_info": "Configure your session in the sidebar, then click Start.",
        "start_session": "Start Session",
        "current_char": "Current: {}",
    },
    "zh": {
        "title": "✨ 无限面试系统 ✨",
        "subtitle": "沉浸式角色扮演面试体验",
        "sidebar_config": "⚙️ 配置",
        "lang_select": "🌐 语言 / Language",
        "jd_section": "📄 职位描述 (JD)",
        "jd_label": "输入职位描述",
        "jd_help": "粘贴职位描述以生成相关问题",
        "kb_section": "📚 知识库",
        "kb_upload": "上传学习资料",
        "kb_help": "上传文本文件以生成面试题",
        "kb_process": "📥 处理知识库",
        "kb_success": "从知识库生成了 {} 道题目！",
        "card_section": "🎭 角色卡片",
        "card_upload": "上传角色卡",
        "card_help": "上传 SillyTavern 格式的角色卡 (JSON 或 PNG)",
        "card_load": "📥 召唤角色",
        "card_success": "已召唤: {} ({})",
        "card_error": "召唤失败: {}",
        "start_button": "🎬 进入世界",
        "submit_answer": "📤 提交回答",
        "next_question": "⏭️ 下一题",
        "feedback_title": "审判",
        "correct": "✅ 正确",
        "incorrect": "❌ 错误",
        "partial": "⚠️ 不完全正确",
        "score": "得分",
        "loading": "世界正在重构...",
        "no_questions": "虚空之中空无一物。（请通过知识库添加题目）",
        "intro_placeholder": "输入你的回应...",
        "answer_placeholder": "说出你的答案...",
        "retry": "重试",
        "hints_title": "💡 提示与类比",
        "keywords": "关键词:",
        "eli5": "通俗解释:",
        "tech_q": "技术问题:",
        "std_ans": "标准答案:",
        "config_info": "请在侧边栏配置，然后点击开始。",
        "start_session": "开始会话",
        "current_char": "当前角色: {}",
    }
}

def t(key: str) -> str:
    """Get translated text."""
    lang = st.session_state.get("language", "zh")
    return TRANSLATIONS.get(lang, TRANSLATIONS["zh"]).get(key, key)


# ============================================================================
# Styling
# ============================================================================

def load_custom_css():
    """Load CSS based on current stage."""
    
    stage = st.session_state.get("stage", "config")
    
    if stage == "config":
        # === Title Screen Style (Reference Image) ===
        css = """
        <style>
        .stApp {
            background: #ffffff;
            color: #000000;
        }
        
        /* Center content vertically and horizontally using Flexbox */
        .main .block-container {
            max-width: 100%;
            height: 90vh; /* Use viewport height */
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            padding-top: 0;
            padding-bottom: 0;
            text-align: center;
            background: transparent;
            box-shadow: none;
            border: none;
            position: static;
            max-height: none;
            overflow: hidden;
        }
        
        /* Hide default elements we don't want */
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        
        /* Button Style - Black Block - Larger */
        .stButton > button {
            background-color: #000000;
            color: #ffffff;
            border: none;
            border-radius: 0;
            padding: 20px 80px;
            font-size: 28px;
            font-weight: 900;
            letter-spacing: 3px;
            transition: all 0.3s;
            margin-top: 40px;
            text-transform: uppercase;
        }
        
        .stButton > button:hover {
            background-color: #333333;
            color: #ffffff;
            transform: scale(1.05);
            box-shadow: 0 15px 30px rgba(0,0,0,0.3);
        }
        
        /* Ensure button container doesn't take full width */
        .stButton {
            width: auto !important;
            display: inline-block;
        }
        
        /* Input fields if any */
        .stTextInput > div > div > input {
            text-align: center;
            border: 2px solid #000;
            border-radius: 0;
        }
        </style>
        """
    else:
        # === Visual Novel Style (Interview Mode) ===
        
        # Load background image
        bg_image_data = ""
        try:
            bg_path = Path("data/emperor_background.jpg")
            if bg_path.exists():
                with open(bg_path, "rb") as f:
                    import base64
                    encoded = base64.b64encode(f.read()).decode()
                    bg_image_data = f"background-image: url('data:image/jpeg;base64,{encoded}');"
        except Exception as e:
            print(f"Background image not loaded: {e}")
            
        css = f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Noto+Serif+SC:wght@500;700&family=Noto+Sans+SC:wght@400;900&display=swap');

        /* Global Styles */
        .stApp {{
            background: #f5f5f5;
            {bg_image_data}
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: #1a1a1a;
            font-family: 'Nsongoto Serif SC', 'Songti SC', 'SimSun', serif;
        }}
        
        /* Screentone Overlay (网点纸效果) */
        .stApp::after {{
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100vw;
            height: 100vh;
            background-image: radial-gradient(#000 1px, transparent 1px);
            background-size: 4px 4px;
            opacity: 0.15;
            z-index: -1;
            pointer-events: none;
        }}
        
        /* 隐藏默认元素 */
        #MainMenu {{visibility: hidden;}}
        footer {{visibility: hidden;}}
        header {{visibility: hidden;}}

        /* === 游戏面板 (Game Panel) === */
        .main .block-container {{
            position: fixed;
            bottom: 5vh;
            left: 10vw;
            right: 10vw;
            width: 80vw;
            max-width: 1200px;
            margin: 0 auto;
            
            background: rgba(255, 255, 255, 0.98);
            border: 4px solid #000000;
            
            /* 双层边框效果 */
            box-shadow: 
                0 0 0 4px #ffffff, /* 白间隙 */
                0 0 0 8px #000000, /* 外黑框 */
                0 20px 50px rgba(0,0,0,0.5); /* 阴影 */
            
            padding: 2rem 5rem; /* 减少上下内边距 */
            z-index: 10;
            height: 35vh; /* 固定高度，缩小面板 */
            overflow-y: auto;
            border-radius: 2px;
        }}
        
        /* === 独立名字栏 (Nameplate) === */
        .character-nameplate {{
            position: fixed;
            bottom: 38vh; /* 调整位置：5vh (bottom) + 35vh (height) - 2vh (overlap) */
            left: 12vw;
            z-index: 12;
            
            background: #000000;
            color: #ffffff;
            padding: 10px 50px;
            
            font-family: 'Noto Sans SC', sans-serif;
            font-weight: 900;
            font-size: 32px; /* 名字也加大 */
            letter-spacing: 3px;
            
            /* 倾斜设计 */
            transform: skew(-15deg);
            border: 3px solid #ffffff;
            box-shadow: 
                0 0 0 3px #000000,
                5px 5px 15px rgba(0,0,0,0.4);
        }}
        
        .character-nameplate span {{
            display: block;
            transform: skew(15deg); /* 文字回正 */
        }}

        /* === 系统消息 (Technical Protocol) === */
        .system-protocol {{
            background: #000000;
            color: #ffffff;
            padding: 20px;
            margin-bottom: 25px;
            font-family: 'Noto Sans SC', sans-serif;
            border: 1px solid #000;
            box-shadow: 0 4px 10px rgba(0,0,0,0.3);
        }}

        /* === 角色台词 (Dialogue) === */
        .dialogue-text {{
            font-family: 'Songti SC', 'SimSun', 'Noto Serif SC', serif; /* 优先宋体 */
            font-size: 26px; /* 调小字号 */
            line-height: 1.6;
            color: #000000;
            font-weight: 500; /* 不加粗，宋体本身较细，500适中 */
            text-align: justify;
            text-shadow: none; /* 去掉阴影，保持干净 */
        }}
        
        /* 强制统一内部所有元素样式，解决Markdown导致的字体不一致 */
        .dialogue-text * {{
            font-size: inherit !important;
            font-weight: inherit !important;
            font-family: inherit !important;
            line-height: inherit !important;
            color: inherit !important;
            margin: 0 !important;
            padding: 0 !important;
        }}
        
        /* 光标闪烁效果 */
        .cursor-blink {{
            display: inline-block;
            width: 14px;
            height: 34px;
            background-color: #000000;
            margin-left: 8px;
            animation: blink 1s step-end infinite;
            vertical-align: middle;
        }}
        
        @keyframes blink {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0; }}
        }}
        
        /* 下一步按钮容器 (覆盖Streamlit默认按钮样式) */
        .next-button-container button {{
            position: fixed !important;
            bottom: 7vh !important;
            right: 12vw !important;
            left: auto !important; /* 强制取消左对齐 */
            background: transparent !important;
            border: none !important;
            color: #000000 !important;
            font-size: 40px !important;
            line-height: 1 !important;
            padding: 0 !important;
            margin: 0 !important;
            z-index: 15 !important;
            box-shadow: none !important;
            animation: bounce 1s infinite;
        }}
        
        .next-button-container button:hover {{
            color: #333333 !important;
            background: transparent !important;
            transform: translateY(2px);
        }}
        
        .next-button-container button:active {{
            background: transparent !important;
            color: #000000 !important;
        }}
        
        @keyframes bounce {{
            0%, 100% {{ transform: translateY(0); }}
            50% {{ transform: translateY(6px); }}
        }}

        /* === 输入框与按钮 === */
        .stTextInput input {{
            background: transparent;
            border: none;
            border-bottom: 3px solid #000000;
            border-radius: 0;
            color: #000000;
            font-family: 'Noto Serif SC', serif;
            font-size: 24px;
            padding: 12px 0;
            font-weight: bold;
        }}
        
        .stTextInput input:focus {{
            box-shadow: none;
            border-bottom: 3px solid #000000;
        }}
        
        .stButton > button {{
            background: #000000;
            color: #ffffff;
            border: none;
            border-radius: 0;
            font-family: 'Noto Sans SC', sans-serif;
            font-weight: bold;
            padding: 10px 30px;
            margin-top: 10px;
        }}
        
        .stButton > button:hover {{
            background: #333333;
        }}
        
        /* 侧边栏 */
        section[data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 3px solid #000000;
        }}
        
        /* === 修复 st.toast 被遮挡的问题 (Top Level z-index) === */
        div[data-testid="stToastContainer"] {{
            z-index: 999999 !important;
        }}
        
        div[data-testid="stToast"] {{
            z-index: 999999 !important;
            background: rgba(255, 255, 255, 0.95);
            border: 2px solid #000;
            box-shadow: 5px 5px 15px rgba(0,0,0,0.5);
            font-family: 'Noto Sans SC', sans-serif;
            color: #000;
        }}
        </style>
        """
    
    st.markdown(css, unsafe_allow_html=True)



# ============================================================================
# Session State
# ============================================================================

def initialize_session_state():
    """Initialize Streamlit session state."""
    if "stage" not in st.session_state:
        st.session_state.stage = "config"  # config, intro, interview

    if "language" not in st.session_state:
        st.session_state.language = "zh"  # Default to Chinese

    if "knowledge_engine" not in st.session_state:
        from aegis_isle.interview.knowledge_engine import KnowledgeEngine
        st.session_state.knowledge_engine = KnowledgeEngine()
    else:
        # Force reload knowledge_engine module to pick up hotfixes
        import importlib
        import sys
        if 'aegis_isle.interview.knowledge_engine' in sys.modules:
            importlib.reload(sys.modules['aegis_isle.interview.knowledge_engine'])
            from aegis_isle.interview.knowledge_engine import KnowledgeEngine
            st.session_state.knowledge_engine = KnowledgeEngine()
            print("🔄 KnowledgeEngine module reloaded and instance recreated.")

    if "persona_manager" not in st.session_state:
        st.session_state.persona_manager = PersonaManager()

    if "generator" not in st.session_state:
        from aegis_isle.interview.generator import Generator
        st.session_state.generator = Generator()
    else:
        # Force reload generator module to pick up hotfixes
        import importlib
        import sys
        if 'aegis_isle.interview.generator' in sys.modules:
            importlib.reload(sys.modules['aegis_isle.interview.generator'])
            from aegis_isle.interview.generator import Generator
            st.session_state.generator = Generator()
            print("🔄 Generator module reloaded and instance recreated.")
    
    if "story_manager" not in st.session_state:
        st.session_state.story_manager = StoryManager()

    if "current_persona" not in st.session_state:
        # Default to Gojo if no card uploaded
        st.session_state.current_persona = st.session_state.persona_manager.get_default_persona()

    if "emperor_persona" not in st.session_state:
        st.session_state.emperor_persona = st.session_state.current_persona
        st.session_state.tutor_personas = []
        st.session_state.current_tutor = st.session_state.current_persona

    if "current_question" not in st.session_state:
        st.session_state.current_question = None

    if "polyphonic_question" not in st.session_state:
        st.session_state.polyphonic_question = None

    if "feedback_data" not in st.session_state:
        st.session_state.feedback_data = None

    if "jd_context" not in st.session_state:
        st.session_state.jd_context = ""
    
    # Track answered questions in current session to prevent immediate repetition
    if "answered_question_ids" not in st.session_state:
        st.session_state.answered_question_ids = []
    
    # Track if we should show a story node
    if "pending_story_node" not in st.session_state:
        st.session_state.pending_story_node = None
    
    # Emperor's Satisfaction Score (0-100)
    if "satisfaction_score" not in st.session_state:
        st.session_state.satisfaction_score = 50  # Start at 50%


# ============================================================================
# Logic Functions
# ============================================================================

async def generate_new_question():
    """Fetch next question and generate polyphonic version."""
    # Check if we should trigger a story node first
    if st.session_state.pending_story_node:
        return  # Story node will be rendered instead
    
    # Get recently answered IDs to avoid immediate repetition
    recent_ids = st.session_state.answered_question_ids[-3:] if len(st.session_state.answered_question_ids) > 0 else []
    
    # Get next question with exclusions
    question = st.session_state.knowledge_engine.get_next_question(exclude_ids=recent_ids)
    
    if not question:
        st.warning(t("no_questions"))
        return

    st.session_state.current_question = question
    
    import random
    if st.session_state.tutor_personas:
        st.session_state.current_tutor = random.choice(st.session_state.tutor_personas)
    else:
        st.session_state.current_tutor = st.session_state.current_persona

    with st.spinner(t("loading")):
        if hasattr(st.session_state.generator, 'generate_dual_question_interaction'):
            poly_q = await st.session_state.generator.generate_dual_question_interaction(
                st.session_state.emperor_persona,
                st.session_state.current_tutor,
                question,
                st.session_state.jd_context,
                language=st.session_state.language
            )
        else:
            poly_q = await st.session_state.generator.generate_question_interaction(
                st.session_state.emperor_persona,
                question,
                st.session_state.jd_context,
                language=st.session_state.language
            )
        st.session_state.polyphonic_question = poly_q
        st.session_state.feedback_data = None  # Reset feedback


async def submit_answer(user_answer: str):
    """Process user answer and generate feedback."""
    if not user_answer.strip():
        return

    with st.spinner(t("loading")):
        if hasattr(st.session_state.generator, 'generate_dual_feedback'):
            feedback = await st.session_state.generator.generate_dual_feedback(
                st.session_state.emperor_persona,
                st.session_state.current_tutor,
                st.session_state.current_question,
                user_answer,
                {},
                language=st.session_state.language
            )
        else:
            feedback = await st.session_state.generator.generate_feedback(
                st.session_state.emperor_persona,
                st.session_state.current_question,
                user_answer,
                {},
                language=st.session_state.language
            )
        st.session_state.feedback_data = feedback
        
        # Update progress
        verdict_status = feedback.get("verdict", {}).get("status")
        is_correct = verdict_status == "correct"
        is_partial = verdict_status == "partial"
        
        st.session_state.knowledge_engine.update_progress(
            st.session_state.current_question.id,
            is_correct
        )
        
        # Update Emperor's Satisfaction Score
        if is_correct:
            st.session_state.satisfaction_score = min(100, st.session_state.satisfaction_score + 15)
        elif is_partial:
            st.session_state.satisfaction_score = min(100, st.session_state.satisfaction_score + 5)
        else:
            st.session_state.satisfaction_score = max(0, st.session_state.satisfaction_score - 10)
        
        # Track this question as answered
        st.session_state.answered_question_ids.append(st.session_state.current_question.id)
        
        # Record answer in story manager
        st.session_state.story_manager.record_answer(is_correct)
        
        # 发送事件到 Aegis LifeEventBus (由 Agent 甲连线)
        try:
            import httpx
            import asyncio
            def send_event_bg():
                event_data = {
                    "source": "love_and_code",
                    "event_type": "interview_practice",
                    "action": "answer_submitted",
                    "content": f"做了面试题: {st.session_state.current_question.question[:50]}...\n回答是否正确: {is_correct}"
                }
                try:
                    # 假定 Aegis 运行在 8001 端口
                    httpx.post("http://127.0.0.1:8001/v1/diary/event", json=event_data, timeout=3.0)
                except Exception as e:
                    print(f"[Love&Code] 无法连接到 Aegis EventBus: {e}")
            
            # 在后台线程发送，不阻塞面试 UI
            import threading
            threading.Thread(target=send_event_bg, daemon=True).start()
        except Exception as e:
            print(f"Error starting event thread: {e}")
        
        # Check if we should trigger a story node
        all_questions = st.session_state.knowledge_engine.questions.values()
        box_levels = [q.review_box for q in all_questions]
        story_trigger = st.session_state.story_manager.check_box_milestone(box_levels)
        
        if story_trigger:
            st.session_state.pending_story_node = story_trigger


async def ingest_kb(file):
    """Ingest knowledge base from uploaded file."""
    with st.spinner(t("processing")):
        text_content = file.read().decode('utf-8')
        questions = await st.session_state.knowledge_engine.ingest_data(text_content, st.session_state.jd_context)
        st.success(f"{t('success_kb')} {len(questions)} {t('questions_generated')}")


def load_emperor_test():
    """Load Emperor test scenario without file upload."""
    import json
    from pathlib import Path
    
    try:
        # Load emperor card
        card_path = Path("data/emperor_card.json")
        if not card_path.exists():
            st.error("测试数据不存在，请先运行: python create_emperor_test.py")
            return False
        
        with open(card_path, 'r', encoding='utf-8') as f:
            card_data = json.load(f)
        
        # Create Persona from card
        from aegis_isle.interview.persona_manager import Persona
        
        emperor = Persona(
            name=card_data.get("name", "人类帝皇"),
            role="人类之主，黄金王座的统治者",
            description=card_data.get("description", ""),
            personality=card_data.get("personality", ""),
            first_message=card_data.get("first_mes", ""),
            example_messages=card_data.get("mes_example", ""),
            scenario=card_data.get("scenario", ""),
            character_book=card_data.get("character_book", {}),
            avatar_path=None
        )
        
        st.session_state.emperor_persona = emperor
        st.session_state.current_persona = emperor # Fallback
        
        # Load the 3 tutors from PersonaManager
        pm = st.session_state.persona_manager
        # Ensure default personas are there
        tutors = []
        if "gojo" in pm.personas: tutors.append(pm.personas["gojo"])
        if "sukuna" in pm.personas: tutors.append(pm.personas["sukuna"])
        if "nanami" in pm.personas: tutors.append(pm.personas["nanami"])
        st.session_state.tutor_personas = tutors
        
        # Load question database
        db_path = Path("data/emperor_test_db.json")
        if not db_path.exists():
            st.error("题库不存在，请先运行: python create_emperor_test.py")
            return False
        
        with open(db_path, 'r', encoding='utf-8') as f:
            db_data = json.load(f)
        
        # Load questions into knowledge engine
        from aegis_isle.interview.knowledge_engine import Question
        
        st.session_state.knowledge_engine.questions = {}
        for qid, qdata in db_data.get("questions", {}).items():
            question = Question(**qdata)
            st.session_state.knowledge_engine.questions[qid] = question
        
        st.session_state.knowledge_engine.save_database()
        
        return True
    
    except Exception as e:
        st.error(f"加载失败: {e}")
        return False

async def load_default_scenario():
    """Load default scenario files (JD, KB, Persona)."""
    from pathlib import Path
    import json
    from aegis_isle.interview.persona_manager import Persona

    default_dir = Path("default")
    jd_path = default_dir / "jd.txt"
    kb_path = default_dir / "llm.md"
    card_path = default_dir / "bigE.json"

    if not (jd_path.exists() and kb_path.exists() and card_path.exists()):
        st.error("Default files missing in 'default/' directory.")
        return False

    try:
        with st.spinner("Loading default scenario..."):
            # 1. Load JD
            with open(jd_path, "r", encoding="utf-8") as f:
                st.session_state.jd_context = f.read()
            
            # 2. Load Knowledge Base
            with open(kb_path, "r", encoding="utf-8") as f:
                kb_content = f.read()
                # Clear existing questions to ensure clean slate
                st.session_state.knowledge_engine.questions = {}
                questions = await st.session_state.knowledge_engine.ingest_data(kb_content, st.session_state.jd_context, language=st.session_state.language)
            
            # 3. Load Persona
            with open(card_path, "r", encoding="utf-8") as f:
                card_data = json.load(f)
                # Handle SillyTavern card format if needed, or simple JSON
                # Assuming bigE.json is a simple JSON or compatible format
                # If it's a SillyTavern card, we might need more complex parsing logic
                # For now, let's assume it matches the Persona structure or use the manager
                
                # Check if it's a V2 card (spec_version) or simple dict
                if "spec" in card_data and "data" in card_data: # V2
                     data = card_data["data"]
                     persona = Persona(
                        name=data.get("name", "Unknown"),
                        role="Interviewer", # Default
                        description=data.get("description", ""),
                        personality=data.get("personality", ""),
                        first_message=data.get("first_mes", ""),
                        example_messages=data.get("mes_example", ""),
                        scenario=data.get("scenario", ""),
                        character_book=data.get("character_book", {}),
                        avatar_path=None
                    )
                else: # Simple or V1
                    persona = Persona(
                        name=card_data.get("name", "Unknown"),
                        role="Interviewer",
                        description=card_data.get("description", ""),
                        personality=card_data.get("personality", ""),
                        first_message=card_data.get("first_mes", ""),
                        example_messages=card_data.get("mes_example", ""),
                        scenario=card_data.get("scenario", ""),
                        character_book=card_data.get("character_book", {}),
                        avatar_path=None
                    )
            
            st.session_state.current_persona = persona
            
            st.success(f"Loaded Default Scenario! Generated {len(questions)} questions.")
            st.session_state.stage = "intro"
            st.rerun()
            return True

    except Exception as e:
        st.error(f"Failed to load default scenario: {e}")
        return False


# ============================================================================
# UI Components
# ============================================================================

def render_sidebar():
    """Render configuration sidebar."""
    with st.sidebar:
        st.title(t("config_title"))
        
        # === Default Scenario Button ===
        if st.button("🚀 加载默认剧本 (Load Default)", type="primary"):
            asyncio.run(load_default_scenario())
        
        st.divider()
        
        # Language Selector
        selected_lang = st.selectbox(
            t("language_selector"),
            options=["zh", "en"],
            format_func=lambda x: "中文" if x == "zh" else "English",
            index=0 if st.session_state.language == "zh" else 1
        )
        if selected_lang != st.session_state.language:
            st.session_state.language = selected_lang
            st.rerun()
        
        st.divider()
        
        # === 快速测试 ===
        st.subheader("⚡ 快速测试" if st.session_state.language == "zh" else "⚡ Quick Test")
        
        if st.button("👑 加载帝皇测试剧本" if st.session_state.language == "zh" else "👑 Load Emperor Test"):
            with st.spinner("正在召唤人类帝皇..." if st.session_state.language == "zh" else "Summoning the Emperor..."):
                success = load_emperor_test()
                if success:
                    st.success("✅ 帝皇测试剧本已加载！" if st.session_state.language == "zh" else "✅ Emperor test loaded!")
                    st.info("📋 已加载 5 道题目\n👑 角色：人类帝皇" if st.session_state.language == "zh" else "📋 5 questions loaded\n👑 Character: Emperor of Mankind")
                else:
                    st.error("❌ 加载失败，请确保测试数据存在" if st.session_state.language == "zh" else "❌ Failed to load test data")
        
        st.divider()
        
        # Job Description
        st.subheader(t("jd_section"))
        st.session_state.jd_context = st.text_area(
            t("jd_label"), 
            value=st.session_state.jd_context,
            height=100,
            help=t("jd_help")
        )

        # Knowledge Base
        st.subheader(t("kb_section"))
        kb_file = st.file_uploader(t("kb_upload"), type=["txt", "md"])
        if kb_file and st.button(t("kb_process")):
            asyncio.run(ingest_kb(kb_file))

        # Character Card
        st.subheader(t("card_section"))
        card_file = st.file_uploader(t("card_upload"), type=["json", "png"])
        if card_file and st.button(t("card_load")):
            try:
                # Save temp
                temp_path = Path(f"temp_{card_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(card_file.read())
                
                # Load
                persona = st.session_state.persona_manager.load_card(temp_path)
                st.session_state.current_persona = persona
                st.success(t("card_success").format(persona.name, persona.role))
                temp_path.unlink()
            except Exception as e:
                st.error(t("card_error").format(e))
        
        st.markdown("---")
        if st.session_state.current_persona:
            st.image(st.session_state.current_persona.avatar_path or "https://placehold.co/200x200?text=Avatar", width=150)
            st.caption(t("current_char").format(st.session_state.current_persona.name))


def stream_text(text, placeholder_container):
    """Stream text with a typewriter effect."""
    import time
    
    # Split text into chunks to simulate natural typing
    # Simple character by character for now
    full_text = ""
    text_placeholder = placeholder_container.empty()
    
    for char in text:
        full_text += char
        # Add cursor
        text_placeholder.markdown(f"""
        <div class="dialogue-text">
            {full_text}<span class="cursor-blink"></span>
        </div>
        """, unsafe_allow_html=True)
        time.sleep(0.02) # Typing speed
        
    # Final state: text only (cursor removed)
    text_placeholder.markdown(f"""
    <div class="dialogue-text">
        {full_text}
    </div>
    """, unsafe_allow_html=True)


def render_intro():
    """Render the Cinematic Intro."""
    persona = st.session_state.current_persona
    
    # Nameplate
    st.markdown(f"""
    <div class="character-nameplate">
        <span>{persona.name}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Dialogue Box Container
    dialogue_container = st.container()
    
    # Stream the intro message if it hasn't been shown yet
    if "intro_shown" not in st.session_state:
        # Clean text: Remove "Name:" prefix if present
        clean_text = persona.first_message.replace(f"{persona.name}：", "").replace(f"{persona.name}:", "")
        # Also handle "人类帝皇" specifically if name doesn't match exactly
        clean_text = clean_text.replace("人类帝皇：", "").replace("人类帝皇:", "")
        
        stream_text(clean_text, dialogue_container)
        st.session_state.intro_shown = True
    else:
        # Show static if already shown
        clean_text = persona.first_message.replace(f"{persona.name}：", "").replace(f"{persona.name}:", "")
        clean_text = clean_text.replace("人类帝皇：", "").replace("人类帝皇:", "")
        
        dialogue_container.markdown(f"""
        <div class="dialogue-text">
            {clean_text}
        </div>
        """, unsafe_allow_html=True)
    
    # Next Button (Clickable Arrow)
    st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
    if st.button("▼", key="intro_next"):
        st.session_state.stage = "interview"
        asyncio.run(generate_new_question())
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def render_story_node():
    """Render a story node (cinematic moment)."""
    story_trigger = st.session_state.pending_story_node
    
    if not story_trigger:
        return False
    
    # Get trigger description
    trigger_info = st.session_state.story_manager.triggers.get(story_trigger)
    
    # Generate story content if not already generated
    if "current_story_content" not in st.session_state:
         # ... (Generation logic same as before) ...
         # For brevity, assuming generation is fast or handled elsewhere
         # Re-implementing generation here for correctness:
        success_rate = st.session_state.story_manager.get_success_rate()
        if "box_1" in story_trigger:
            node_type = "node_a"
            title = "🧬 初次觉醒"
        elif "box_3" in story_trigger:
            node_type = "node_b"  
            title = "⚔️ 晋升试炼"
        else:
            node_type = "mastery"
            title = "👑 荣誉时刻"
            
        with st.spinner("剧情生成中..."):
            story_data = asyncio.run(st.session_state.generator.generate_story_node(
                st.session_state.current_persona,
                node_type,
                success_rate,
                language=st.session_state.language
            ))
            st.session_state.current_story_content = story_data.get("story_content", "...")
            st.session_state.current_story_title = title

    # Render UI
    st.markdown(f"""
    <div class="character-nameplate" style="background: #ffd700; color: #000; border-color: #000;">
        <span>{st.session_state.current_story_title}</span>
    </div>
    """, unsafe_allow_html=True)
    
    dialogue_container = st.container()
    
    # Stream text
    if "story_text_shown" not in st.session_state:
        stream_text(st.session_state.current_story_content, dialogue_container)
        st.session_state.story_text_shown = True
    else:
        dialogue_container.markdown(f"""
        <div class="dialogue-text">
            {st.session_state.current_story_content}
        </div>
        """, unsafe_allow_html=True)
    
    # Next Button for Story Node
    st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
    if st.button("▼", key="continue_from_story"):
        st.session_state.pending_story_node = None
        # Clear story state
        del st.session_state.current_story_content
        del st.session_state.current_story_title
        del st.session_state.story_text_shown
        
        asyncio.run(generate_new_question())
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)
    
    return True


def render_interview():
    """Render the Interview Loop with Advanced Visual Novel UI."""
    # Check if we should show a story node first
    if st.session_state.pending_story_node:
        if render_story_node():
            return  # Story node is being displayed
    
    poly_q = st.session_state.polyphonic_question
    
    if not poly_q:
        st.error("No question generated.")
        if st.button(t("retry")):
            asyncio.run(generate_new_question())
            st.rerun()
        return

    # === Menu & Settings Buttons (Top Left) - DISABLED ===
    # (Removed to fix syntax error - can be re-added with proper JavaScript injection later)


    # === Emperor's Satisfaction Progress Bar (Top Right) ===
    satisfaction = st.session_state.satisfaction_score
    st.markdown(f"""
    <div style="position: fixed; top: 20px; right: 20px; z-index: 9999; background: rgba(255,255,255,0.95); border: 4px solid #000; padding: 15px 20px; box-shadow: 0 5px 15px rgba(0,0,0,0.3);">
        <div style="font-family: 'Noto Sans SC', sans-serif; font-weight: 900; font-size: 16px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <span>⚡ 帝皇的满意度</span>
            <span style="margin-left: 15px;">{satisfaction}%</span>
        </div>
        <div style="width: 250px; height: 30px; border: 3px solid #000; background: #fff; position: relative; overflow: hidden;">
            <div style="position: absolute; width: {satisfaction}%; height: 100%; background: repeating-linear-gradient(45deg, #000 0px, #000 10px, #fff 10px, #fff 20px); transition: width 0.5s ease;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # === 独立名字栏 (Nameplate) ===
    character_name = st.session_state.current_persona.name
    st.markdown(f"""
    <div class="character-nameplate">
        <span>{character_name}</span>
    </div>
    """, unsafe_allow_html=True)
    
    # === 系统消息 (Technical Protocol) ===
    st.markdown(f"""
    <div class="system-protocol">
        <div style="background: #ffffff; color: #000000; display: inline-block; padding: 2px 8px; font-weight: 900; font-size: 18px; margin-bottom: 8px; transform: skew(-10deg);">技术协议 // PROTOCOL</div>
        <div style="font-size: 22px; line-height: 1.5;">{poly_q.get('original_question', '')}</div>
    </div>
    """, unsafe_allow_html=True)

    # === 角色台词 (Dialogue) ===
    dialogue_container = st.container()
    
    # Check if this specific question's dialogue has been shown
    q_key = f"q_{st.session_state.current_question.id}_shown"
    
    if 'emperor_flavor' in poly_q and 'tutor_flavor' in poly_q:
        emp_text = poly_q['emperor_flavor'].replace("人类帝皇：", "").replace("人类帝皇:", "")
        tut_text = poly_q['tutor_flavor'].replace(f"{st.session_state.current_tutor.name}：", "").replace(f"{st.session_state.current_tutor.name}:", "")
    else:
        raw_dialogue = poly_q.get('lore_flavor', '')
        dialogue_content = raw_dialogue.replace(f"{character_name}：", "").replace(f"{character_name}:", "")
        emp_text = dialogue_content.replace("人类帝皇：", "").replace("人类帝皇:", "")
        tut_text = ""
    
    if q_key not in st.session_state:
        stream_text(emp_text, dialogue_container)
        if tut_text:
            st.toast(f"**【{st.session_state.current_tutor.name}】的小纸条：**\n\n{tut_text}", icon="💬")
        st.session_state[q_key] = True
    else:
        dialogue_container.markdown(f"""
        <div class="dialogue-text">
            {emp_text}
        </div>
        """, unsafe_allow_html=True)

    # Answer Input Area
    if not st.session_state.feedback_data:
        st.markdown("---")
        # Custom Input Box Style matching reference
        st.markdown("""
        <style>
        .stTextInput input {
            border: 2px solid #000 !important;
            background: #ffffff !important;
            color: #000 !important;
            padding: 15px !important;
            font-size: 18px !important;
            height: 56px !important;
            box-sizing: border-box !important;
        }
        
        .stButton button {
            background: #000 !important;
            color: #fff !important;
            border: none !important;
            height: 56px !important;
            font-size: 18px !important;
            font-weight: bold !important;
            padding: 0 30px !important;
            margin-top: -2px !important; /* 上移对齐输入框 */
        }
        </style>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([4, 1])
        with col1:
            user_answer = st.text_input("Answer", key="answer_input", label_visibility="collapsed", placeholder="在此输入你的回答...")
        with col2:
            if st.button("提交回答", use_container_width=True):
                asyncio.run(submit_answer(user_answer))
                st.rerun()

        # === 提示信息 (Hints) - Only show when no feedback ===
        # Tech Hint (Professional)
        if poly_q.get('tech_hint'):
            st.markdown(f"""
            <div style="margin-top: 20px; border-left: 4px solid #000; padding-left: 15px; color: #444; font-family: 'Noto Sans SC', sans-serif; font-size: 18px;">
                <strong>💡 技术提示：</strong> {poly_q.get('tech_hint')}
            </div>
            """, unsafe_allow_html=True)
            
        # ELI5 Hint (Warhammer 40k Servitor Style)
        if poly_q.get('eli5_hint'):
            st.markdown(f"""
            <div style="margin-top: 10px; border-left: 4px solid #666; padding-left: 15px; color: #666; font-family: 'Noto Sans SC', sans-serif; font-size: 18px; font-style: italic;">
                <strong>🤖 机仆注视：</strong> {poly_q.get('eli5_hint')}
            </div>
            """, unsafe_allow_html=True)
    else:
        # Feedback Display - Three-Fold Judgment
        fb = st.session_state.feedback_data
        verdict_data = fb.get("verdict", {})
        status = verdict_data.get("status", "partial")
        
        # Visual Effects based on answer correctness
        if status == "incorrect":
            # Screen Shake Effect (Emperor's Anger)
            st.markdown("""
            <style>
            @keyframes shake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-10px); }
                20%, 40%, 60%, 80% { transform: translateX(10px); }
            }
            .stApp {
                animation: shake 0.5s;
            }
            </style>
            """, unsafe_allow_html=True)
        elif status == "correct":
            # Fireworks Effect (Celebration)
            st.markdown("""
            <style>
            @keyframes firework {
                0% { transform: translate(0, 0) scale(0); opacity: 1; }
                50% { opacity: 1; }
                100% { transform: translate(var(--x), var(--y)) scale(1); opacity: 0; }
            }
            .fireworks {
                position: fixed;
                top: 50%;
                left: 50%;
                width: 100%;
                height: 100%;
                pointer-events: none;
                z-index: 9998;
            }
            .firework {
                position: absolute;
                font-size: 40px;
                animation: firework 1.5s ease-out forwards;
            }
            </style>
            <div class="fireworks">
                <div class="firework" style="--x: -200px; --y: -200px; animation-delay: 0s; left: 50%; top: 50%;">🎆</div>
                <div class="firework" style="--x: 200px; --y: -200px; animation-delay: 0.2s; left: 50%; top: 50%;">✨</div>
                <div class="firework" style="--x: -200px; --y: 200px; animation-delay: 0.4s; left: 50%; top: 50%;">💫</div>
                <div class="firework" style="--x: 200px; --y: 200px; animation-delay: 0.6s; left: 50%; top: 50%;">🎇</div>
                <div class="firework" style="--x: 0px; --y: -250px; animation-delay: 0.3s; left: 50%; top: 50%;">⭐</div>
                <div class="firework" style="--x: 0px; --y: 250px; animation-delay: 0.5s; left: 50%; top: 50%;">🌟</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 1. Emperor & Tutor Verdicts
        verdict_comment = verdict_data.get('comment', '')
        # Clean the verdict comment
        character_name = st.session_state.emperor_persona.name
        verdict_clean = verdict_comment.replace(f"{character_name}：", "").replace(f"{character_name}:", "")
        verdict_clean = verdict_clean.replace("人类帝皇：", "").replace("人类帝皇:", "")
        
        # Merge Tutor's explanation if available
        tut_exp = fb.get("servitor_explanation", "")
        tut_name = st.session_state.current_tutor.name
        tut_clean = tut_exp.replace(f"{tut_name}：", "").replace(f"{tut_name}:", "")
        
        fb_key = f"q_{st.session_state.current_question.id}_fb_shown"
        verdict_container = st.container()
        
        if fb_key not in st.session_state:
            stream_text(verdict_clean, verdict_container)
            if tut_clean:
                st.toast(f"**【{tut_name}】的课后辅导：**\n\n{tut_clean}", icon="💡")
            st.session_state[fb_key] = True
        else:
            verdict_container.markdown(f"""
            <div class="dialogue-text">
                {verdict_clean}
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # 2. Standard Answer (Professional)
        standard_answer = fb.get("standard_answer", "")
        if standard_answer:
            st.markdown(f"""
            <div style="margin-top: 20px; border-left: 4px solid #000; padding-left: 15px; color: #444; font-family: 'Noto Sans SC', sans-serif; font-size: 18px;">
                <strong>💡 标准答案：</strong> {standard_answer}
            </div>
            """, unsafe_allow_html=True)

        # Next Button for Interview
        st.markdown('<div class="next-button-container">', unsafe_allow_html=True)
        if st.button("▼", key="next_question_btn"):
            asyncio.run(generate_new_question())
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


# ============================================================================
# Main App
# ============================================================================

def main():
    st.set_page_config(
        page_title="The Infinite Interview",
        page_icon="🔮",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Initialize session state
    initialize_session_state()
    
    # Load custom CSS (simplified version)
    load_custom_css()
    
    # Render Sidebar
    render_sidebar()
    
    # Main Content Area
    if st.session_state.stage == "config":
        # Title Screen Content
        st.markdown("""
        <div style="text-align: center;">
            <div style="font-family: 'Arial Black', sans-serif; font-size: 80px; font-weight: 900; letter-spacing: 5px; line-height: 1.1; margin-bottom: 20px;">
                THE INFINITE<br>INTERVIEW
            </div>
            <div style="font-family: 'Arial', sans-serif; font-size: 28px; letter-spacing: 15px; color: #000; margin-bottom: 60px; font-weight: bold;">
                无限面试 · 帝皇审判
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Centered Button (No columns needed with flexbox)
        if st.button("接入思维阵列"):
            st.session_state.stage = "intro"
            st.rerun()
            
    elif st.session_state.stage == "intro":
        render_intro()
        
    elif st.session_state.stage == "interview":
        render_interview()

if __name__ == "__main__":
    main()  # Fixed: removed asyncio.run() since main() is not async
