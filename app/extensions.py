from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_cors import CORS

# Экземпляры расширений, которые инициализируются во фабрике приложения
login_manager = LoginManager()
socketio = SocketIO()
cors = CORS()


