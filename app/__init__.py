from flask import Blueprint

app = Blueprint('app', __name__)

from .Servicio import servicio_app
from .Servicio import auth_app