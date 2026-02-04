from contextlib import contextmanager

from data import session as db_session


@contextmanager
def get_db_session():
    """Удобный контекстный менеджер для работы с сессией БД."""
    db_sess = db_session.create_session()
    try:
        yield db_sess
    finally:
        db_sess.close()


