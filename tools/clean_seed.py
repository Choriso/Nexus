"""Удаляет все тестовые данные, оставляя только администратора (id=1)."""
from data.session import global_init, create_session
from config import config
from sqlalchemy import text

global_init(config.DATABASE_URL)
db = create_session()

# Удаляем в порядке от наиболее зависимых к наименее зависимым
db.execute(text("DELETE FROM user_interest_graph_weights"))
db.execute(text("DELETE FROM ai_user_compatibility"))
db.execute(text("DELETE FROM ai_extracted_interests"))
db.execute(text("DELETE FROM ai_user_personality_profiles"))
db.execute(text("DELETE FROM user_behavior_profiles"))
db.execute(text("DELETE FROM user_schwartz_profiles"))
db.execute(text("DELETE FROM knowledge_connections"))
db.execute(text("DELETE FROM knowledge_nodes"))
db.execute(text("DELETE FROM messages"))
db.execute(text("DELETE FROM chats"))
db.execute(text("DELETE FROM favorite_interests"))
db.execute(text("DELETE FROM dynamic_aliases"))
db.execute(text("DELETE FROM users WHERE id != 1"))

db.commit()
print("✅ Все тестовые данные удалены")
db.close()