import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.orm import Session

SqlAlchemyBase = orm.declarative_base()

__factory = None


def global_init(db_conn_string):
    global __factory

    if __factory:
        return

    if not db_conn_string or not db_conn_string.strip():
        raise Exception("Необходимо указать строку подключения к базе данных.")

    db_conn_string = db_conn_string.strip()

    # Проверяем, является ли строка полным URL или просто путем к файлу
    if db_conn_string.startswith(('postgresql://', 'sqlite://', 'mysql://', 'postgres://')):
        # Это уже полный URL с диалектом
        conn_str = db_conn_string
        print(f"Подключение к базе данных по адресу: {conn_str.split('@')[0] if '@' in conn_str else conn_str}")
    else:
        # Это путь к файлу SQLite - добавляем префикс
        conn_str = f'sqlite:///{db_conn_string}?check_same_thread=False'
        print(f"Подключение к SQLite базе данных по адресу: {conn_str}")

    # Создаем движок
    engine = sa.create_engine(conn_str,
                              pool_pre_ping=True,
                              pool_recycle=3600)
    __factory = orm.sessionmaker(bind=engine)

    from . import __all_models

    SqlAlchemyBase.metadata.create_all(engine)


def create_session() -> Session:
    global __factory
    return __factory()
