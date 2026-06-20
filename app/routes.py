from flask import Blueprint, render_template, make_response, request, session
from flask_login import current_user

main_bp = Blueprint('main', __name__)

@main_bp.route("/")
def start():
    return render_template("startScreen.html", current_user=current_user, hide_header=True)

@main_bp.route('/geolocation')
def geolocation():
    return render_template('geolocation_ip.html')
