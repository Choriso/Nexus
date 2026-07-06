# App.py

from app import create_app

from app.extensions import socketio

from config import config



app = create_app()



if __name__ == "__main__":

    print(f"http://{config.FLASK_HOST}:{config.FLASK_PORT}")
    print(f"http://192.168.3.41:{config.FLASK_PORT}")



    socketio.run(

        app,

        host=config.FLASK_HOST,

        port=config.FLASK_PORT,

        debug=config.DEBUG,

    )

