import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import os
from typing import Optional, Tuple, List, Dict, Any
import hashlib

# Настройки
INDEX_FILE = "olympiad_index.pkl"
MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"

@st.cache_resource
def load_model() -> Optional[SentenceTransformer]:
    """Загружает AI-модель один раз и кэширует"""
    try:
        model = SentenceTransformer(MODEL_NAME)
        return model
    except Exception as e:
        st.error(f"❌ Ошибка загрузки модели: {e}")
        return None

@st.cache_data
def load_index() -> Optional[Dict[str, Any]]:
    """Безопасная загрузка индекса"""
    if not os.path.exists(INDEX_FILE):
        return None
    try:
        with open(INDEX_FILE, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"❌ Ошибка загрузки индекса: {e}")
        return None

def check_password(username: str, password: str) -> bool:
    if "credentials" not in st.secrets:
        return username == "admin" and password == "kolbamiha"  # fallback
    
    creds = st.secrets["credentials"]
    if username not in creds:
        return False
    
    password_hash = hashlib.sha256(password.encode()).hexdigest()
    return password_hash == creds[username]

def login():
    """Форма входа"""
    st.markdown("""
        <style>
        .login-container {
            max-width: 400px;
            margin: 100px auto;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<div class="login-container">', unsafe_allow_html=True)
        st.title("🔐 Вход в систему")
        st.markdown("### Поиск олимпиад и конкурсов")
        st.caption("Официальный перечень Министерства просвещения РФ на 2025/26 учебный год")
        
        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("👤 Логин", placeholder="Введите логин")
            password = st.text_input("🔒 Пароль", type="password", placeholder="Введите пароль")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button("Войти", use_container_width=True)
            
            if submit:
                if not username or not password:
                    st.error("❌ Введите логин и пароль")
                elif check_password(username, password):
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.rerun()
                else:
                    st.error("❌ Неверный логин или пароль")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Информация для входа (только для демонстрации)
        with st.expander("ℹ️ Информация для входа (демо)", expanded=False):
            st.markdown("""
            **Для локального тестирования:**
            - Логин: `admin`
            - Пароль: `admin123`
            
            **Для production используйте `.streamlit/secrets.toml`**
            """)

def logout():
    """Выход из системы"""
    st.session_state["logged_in"] = False
    st.session_state["username"] = None
    st.rerun()

def search_olympiad(query: str, top_k: int = 10) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Логика поиска с защитой от пустых результатов"""
    model = load_model()
    if model is None:
        return None, "Модель не загружена"
    
    data = load_index()
    if data is None:
        return None, f"Файл {INDEX_FILE} не найден или поврежден"

    try:
        embeddings = data["embeddings"]
        metadata = data["metadata"]
    except KeyError as e:
        return None, f"Неверная структура индекса: отсутствует ключ {e}"

    # Векторизация запроса
    q_emb = model.encode([query], normalize_embeddings=True)

    # Косинусное сходство
    sims = (embeddings @ q_emb.T).flatten()

    # Берем топ-50 кандидатов для детальной проверки
    top_idx = np.argsort(sims)[::-1][:min(50, len(sims))]

    results = []
    for idx in top_idx:
        sim = float(sims[idx])
        
        # Порог уверенности (семантика)
        if sim < 0.30: 
            continue

        item = metadata[idx]
        
        # Нечеткое сравнение строк (Fuzzy Logic)
        full_text = " ".join([
            item.get('name', ''),
            item.get('organizer', ''),
            item.get('profile', ''),
            item.get('direction', '')
        ])
        fuzzy = fuzz.partial_ratio(query.lower(), full_text.lower())

        # Фильтр по точному совпадению букв
        if fuzzy < 25:
            continue
            
        results.append({
            'name': item.get('name', 'Не указано'),
            'organizer': item.get('organizer', 'Не указано'),
            'direction': item.get('direction', 'Не указано'),
            'profile': item.get('profile', 'Не указано'),
            'level': item.get('level', 'Не указано'),
            'page': item.get('page', '?'),
            'number': item.get('number', ''),
            'score': round(sim * 100, 1),
            'fuzzy': fuzzy
        })
        
        # Если набрали достаточно результатов, выходим
        if len(results) >= top_k:
            break

    return results, None

def main_app():
    """Основное приложение (после авторизации)"""
    st.set_page_config(
        page_title="🔍 Поиск олимпиад",
        layout="wide",
        page_icon="🎓"
    )
    
    # Шапка с кнопкой выхода
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("🏆 Поиск олимпиад и конкурсов")
        st.caption("Официальный перечень Министерства просвещения РФ на 2025/26 учебный год")
    with col2:
        if st.button("🚪 Выход", use_container_width=True):
            logout()
    
    # Приветствие
    st.markdown(f"👤 **Пользователь:** {st.session_state.get('username', 'Гость')}")
    st.markdown("---")

    # Проверка наличия файла индекса при запуске
    if not os.path.exists(INDEX_FILE):
        st.warning(f"⚠️ Файл `{INDEX_FILE}` не найден!")
        st.info("Запустите `indexer.py` для создания базы данных.")
        st.stop()

    # Боковая панель с фильтрами
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Фильтр по уровню
        levels = ["Все уровни", "I уровень", "II уровень", "III уровень", "IV уровень", "Высший уровень", "Не установлен"]
        selected_level = st.selectbox("Уровень мероприятия:", levels)

        # Фильтр по направлению
        directions = ["Все направления", "Наука и образование", "Искусство и культура", "Физическая культура и спорт"]
        selected_direction = st.selectbox("Направление:", directions)

    # Поле поиска
    query = st.text_input(
        "🔎 Поиск по названию, организатору или предмету",
        placeholder="Например: Всероссийская олимпиада по математике",
        help="Введите название полностью или частично"
    )

    if query:
        with st.spinner("🔎 Ищем совпадения..."):
            results, error = search_olympiad(query)
            
            if error:
                st.error(f"⚠️ **Ошибка:** {error}")
            elif not results:
                st.warning("❌ В перечне не найдено. Попробуйте изменить запрос.")
            else:
                # Применяем фильтры
                if selected_level != "Все уровни":
                    results = [r for r in results if r['level'] == selected_level]
                if selected_direction != "Все направления":
                    results = [r for r in results if r['direction'] == selected_direction]
                
                if not results:
                    st.warning("❌ По выбранным фильтрам ничего не найдено.")
                else:
                    st.success(f"✅ Найдено {len(results)} совпадений")
                    
                    for i, r in enumerate(results, 1):
                        with st.expander(f"🏆 {i}. {r['name'][:100]}{'...' if len(r['name']) > 100 else ''}", expanded=(i==1)):
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                if r.get('number'):
                                    st.markdown(f"**🆔 Номер в перечне:** `{r['number']}`")
                                
                                st.markdown("**📋 Название:**")
                                st.write(r['name'])
                                
                                st.markdown("**🏛 Организатор:**")
                                st.write(r['organizer'])
                                
                                st.markdown("**🎯 Направление:**")
                                st.write(r['direction'])
                            
                            with col2:
                                st.markdown("**📌 Профиль/Предмет:**")
                                st.write(r['profile'])
                                
                                st.markdown("**📊 Уровень:**")
                                st.info(r['level'])
                                
                                if r['page'] != '?':
                                    st.markdown(f"**📄 Страница:** {r['page']}")
                            
                            st.caption(f"🤖 AI уверенность: {r['score']}% | 🔤 Совпадение букв: {r['fuzzy']}%")
    else:
        # Статистика при запуске
        data = load_index()
        if data:
            st.markdown("---")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Всего мероприятий в базе", len(data.get("metadata", [])))
            with col2:
                st.metric("Направление", "Все")
            with col3:
                st.metric("Уровень", "Все")

# === ТОЧКА ВХОДА ===
if __name__ == "__main__":
    # Инициализация session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "username" not in st.session_state:
        st.session_state["username"] = None
    
    # Проверяем, авторизован ли пользователь
    if not st.session_state["logged_in"]:
        # Показываем форму входа
        st.set_page_config(
            page_title="🔐 Вход в систему",
            layout="centered",
            page_icon="🔐"
        )
        login()
    else:
        # Показываем основное приложение
        main_app()
