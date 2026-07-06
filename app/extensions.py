from flask_login import LoginManager
from flask_socketio import SocketIO
from flask_cors import CORS
from flask_migrate import Migrate
from data.session import SqlAlchemyBase

migrate = Migrate()
target_metadata = SqlAlchemyBase.metadata
login_manager = LoginManager()
socketio = SocketIO()
cors = CORS()


