import os
from app import create_app

application = create_app()

if __name__ == "__main__":
    from app.extensions import socketio
    from config import config

    socketio.run(application, host=config.FLASK_HOST, port=config.FLASK_PORT, debug=config.DEBUG)
