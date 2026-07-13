from flask_wtf import FlaskForm
from flask_wtf.file import FileRequired
from wtforms import PasswordField, StringField, SubmitField, EmailField, BooleanField, FileField
from wtforms.validators import DataRequired


class RegisterForm(FlaskForm):
    email = EmailField('Логин / email', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    password_again = PasswordField('Повторите пароль', validators=[DataRequired()])
    name = StringField('Никнейм', validators=[DataRequired()])
    information = StringField('О себе', validators=[DataRequired()])
    connection = StringField('Информация для контакта', validators=[DataRequired()])
    submit = SubmitField('Завершить')


class LoginForm(FlaskForm):
    email = EmailField('Email', validators=[DataRequired()])
    password = PasswordField('Пароль', validators=[DataRequired()])
    remember_me = BooleanField('Запомнить меня')
    submit = SubmitField('Войти')