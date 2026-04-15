import streamlit as st
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from rapidfuzz import fuzz
import os
from typing import Optional, Tuple, List, Dict, Any

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
            'number': item.get('number', ''), # Новое поле: номер олимпиады
            'score': round(sim * 100, 1),
            'fuzzy': fuzzy
        })
        
        # Если набрали достаточно результатов, выходим
        if len(results) >= top_k:
            break

    return results, None

# === ИНТЕРФЕЙС ===
st.set_page_config(
    page_title="🔍 Поиск олимпиад", 
    layout="wide",
    page_icon="🎓"
)

st.title("🏆 Поиск олимпиад и конкурсов")
st.caption("Официальный перечень Министерства просвещения РФ на 2025/26 учебный год")

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
        st.success(f"✅ Найдено {len(results)} совпадений")
        
        for i, r in enumerate(results, 1):
            with st.expander(f"🏆 {i}. {r['name'][:100]}{'...' if len(r['name']) > 100 else ''}", expanded=(i==1)):
                col1, col2 = st.columns(2)
                
                with col1:
                    if r.get('number'):
                        st.markdown(f"**🆔 Номер в перечне:** `{r['number']}`")
                    
                    st.markdown(f"**📋 Название:**")
                    st.write(r['name'])
                    
                    st.markdown(f"**🏛 Организатор:**")
                    st.write(r['organizer'])
                    
                    st.markdown(f"**🎯 Направление:**")
                    st.write(r['direction'])
                
                with col2:
                    st.markdown(f"**📌 Профиль/Предмет:**")
                    st.write(r['profile'])
                    
                    st.markdown(f"**📊 Уровень:**")
                    st.info(r['level']) # Выделяем уровень цветом
                    
                    if r['page'] != '?':
                        st.markdown(f"**📄 Страница:** {r['page']}")
                
                st.caption(f"🤖 AI уверенность: {r['score']}% | 🔤 Совпадение букв: {r['fuzzy']}%")

else:
    # Статистика при запуске
    data = load_index()
    if data:
        st.markdown("---")
        st.metric("Всего мероприятий в базе", len(data.get("metadata", [])))
