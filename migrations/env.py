import logging
from logging.config import fileConfig
from flask import current_app
from alembic import context
import sqlalchemy as sa

# Это объект конфигурации Alembic
config = context.config

# Настройка логирования
fileConfig(config.config_file_name)
logger = logging.getLogger('alembic.env')

def get_engine_url():
    """Получает строку подключения напрямую из конфигурации Flask."""
    url = current_app.config.get("SQLALCHEMY_DATABASE_URI")
    return str(url).replace('%', '%%')

# Устанавливаем URL для Alembic
config.set_main_option('sqlalchemy.url', get_engine_url())

def get_metadata():
    """Возвращает метаданные моделей из вашей кастомной структуры."""
    from app.extensions import target_metadata
    return target_metadata

def run_migrations_offline():
    """Запуск миграций в offline-режиме."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url, target_metadata=get_metadata(), literal_binds=True
    )

    with context.begin_transaction():
        context.run_migrations()

def run_migrations_online():
    """Запуск миграций в online-режиме."""
    def process_revision_directives(context, revision, directives):
        if getattr(config.cmd_opts, 'autogenerate', False):
            script = directives[0]
            if script.upgrade_ops.is_empty():
                directives[:] = []
                logger.info('No changes in schema detected.')

    conf_args = current_app.extensions['migrate'].configure_args
    if conf_args.get("process_revision_directives") is None:
        conf_args["process_revision_directives"] = process_revision_directives

    # Создаем connectable-движок напрямую, используя URL из приложения
    connectable = sa.create_engine(current_app.config["SQLALCHEMY_DATABASE_URI"])

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=get_metadata(),
            **conf_args
        )

        with context.begin_transaction():
            context.run_migrations()

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()