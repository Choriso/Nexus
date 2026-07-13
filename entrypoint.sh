#!/bin/bash
set -e

if [ "$FLASK_ENV" = "production" ]; then
    echo "Running database migrations..."
    flask db upgrade 2>/dev/null || echo "Migrations skipped (no Flask-Migrate heads)"

    echo "Starting gunicorn with eventlet workers..."
    exec gunicorn -k eventlet -w 1 \
        --bind 0.0.0.0:8000 \
        --worker-connections 1000 \
        --access-logfile - \
        --error-logfile - \
        --log-level "${GUNICORN_LOG_LEVEL:-info}" \
        --timeout 120 \
        wsgi:application
else
    echo "Development mode - starting Flask dev server..."
    exec python wsgi.py
fi
