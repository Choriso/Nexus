# App.py
import os
from app import create_app
from app.extensions import socketio

app = create_app()

if __name__ == "__main__":
    host = os.environ.get("FLASK_HOST", "127.0.0.1")
    port = int(os.environ.get("FLASK_PORT", "3000"))
    debug_mode = os.environ.get("FLASK_ENV") == "development"

    print(f"🚀 Nexus запущен на http://{host}:{port}")

    socketio.run(app, host=host, port=port, debug=debug_mode)