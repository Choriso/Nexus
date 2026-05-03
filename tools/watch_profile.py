import time
import os
from dotenv import load_dotenv
from data import session as db_session
from app.ai.models import UserPersonalityProfile

load_dotenv()
db_session.global_init(os.environ.get("DATABASE_URL", "sqlite:///db/blogs.db"))


def watch_profile(user_id):
    print("\033[2J\033[H", end="", flush=True)  # Очистка экрана один раз при старте

    try:
        while True:
            sess = db_session.create_session()
            profile = sess.query(UserPersonalityProfile).filter_by(user_id=user_id).first()
            sess.close()

            lines = []
            lines.append(f"🎯 ДАННЫЕ ИЗ БАЗЫ (User #{user_id}):")
            lines.append("=" * 45)

            if profile:
                traits = {
                    "Открытость": profile.openness,
                    "Добросовестность": profile.conscientiousness,
                    "Экстраверсия": profile.extraversion,
                    "Дружелюбие": profile.agreeableness,
                    "Невротизм": profile.neuroticism
                }
                for name, val in traits.items():
                    bar = "█" * int(val * 25)
                    lines.append(f"{name:<18} | {val:.4f} {bar}")
            else:
                lines.append("Нет данных...")

            lines.append("=" * 45)
            lines.append(f"🕒 Обновлено: {time.strftime('%H:%M:%S')}")

            # \033[H — курсор в начало
            # \033[2K — очистка текущей строки перед печатью
            output = "\033[H"
            for line in lines:
                output += f"\033[2K{line}\n"

            print(output, end="", flush=True)
            time.sleep(1)

    except KeyboardInterrupt:
        print("\n\nМониторинг остановлен.")


if __name__ == "__main__":
    watch_profile(3)