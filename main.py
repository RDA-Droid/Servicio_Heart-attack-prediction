from flask import Flask
from waitress import serve
from app import servicio_app, auth_app

app = Flask(__name__)

app.register_blueprint(servicio_app)
app.register_blueprint(auth_app, url_prefix='/auth')

if __name__ == '__main__':
    serve(app, host='127.0.0.1', port=5000)
