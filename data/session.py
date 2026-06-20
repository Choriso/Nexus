import sqlalchemy as sa
import sqlalchemy.orm as orm
from sqlalchemy.orm import Session

SqlAlchemyBase = orm.declarative_base()
__factory = None


def global_init(db_conn_string: str) -> None:
    """
    Инициализирует глобальную фабрику сессий SQLAlchemy.

    Args:
        db_conn_string (str): Строка подключения к базе данных.
    
    Raises:
        Exception: Если строка подключения не указана или пуста.

    Returns:
        None
    """
    global __factory

    if __factory:
        return

    if not db_conn_string or not db_conn_string.strip():
        raise Exception("Необходимо указать строку подключения к базе данных.")

    db_conn_string = db_conn_string.strip()

    if db_conn_string.startswith(('postgresql://', 'sqlite://', 'mysql://', 'postgres://')):
        conn_str = db_conn_string
        print(f"Подключение к базе данных по адресу: {conn_str.split('@')[0] if '@' in conn_str else conn_str}")
    else:
        conn_str = f'sqlite:///{db_conn_string}?check_same_thread=False'
        print(f"Подключение к SQLite базе данных по адресу: {conn_str}")

    engine = sa.create_engine(
        conn_str,
        pool_pre_ping=True,
        pool_recycle=3600
    )
    __factory = orm.sessionmaker(bind=engine)

    from . import __all_models

    SqlAlchemyBase.metadata.create_all(engine)


def create_session() -> Session:
    """
    Создаёт и возвращает новую сессию SQLAlchemy.

    Returns:
        Session: Новый экземпляр сессии SQLAlchemy.
    """
    global __factory
    return __factory()
