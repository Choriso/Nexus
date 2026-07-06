"""
Legacy launcher kept for compatibility.

Chat routes are now implemented only through blueprint `app/chat.py`.
Run this module if you still need a top-level script entrypoint.
"""

from app import create_app
from app.extensions import socketio
from config import config

app = create_app()

if __name__ == "__main__":
    socketio.run(
        app,
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.DEBUG,
    )
