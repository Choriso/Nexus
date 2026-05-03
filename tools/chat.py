"""
Legacy launcher kept for compatibility.

Chat routes are now implemented only through blueprint `app/chat.py`.
Run this module if you still need a top-level script entrypoint.
"""

import os

from app import create_app
from app.extensions import socketio

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "3000"))
    debug_mode = os.environ.get("FLASK_ENV") == "development"
    socketio.run(app, host=host, port=port, debug=debug_mode)
