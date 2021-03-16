from django.contrib.auth.models import User
from django.contrib.auth.backends import BaseBackend

from sqlalchemy import create_engine
import pandas as pd
import hashlib


def encode_pass(p):
    return hashlib.md5(str.encode(p)).digest()

def check_login(user, pwd):
    login = False
    db = get_sql_connection()
    try:
        df = pd.read_sql(f'select * from user_login where username="{user}"', db)
        login = encode_pass(pwd) == df['password'].iloc[0]
    except Exception as e:
        pass
    return login

def get_sql_connection():
    db = create_engine(
        "mysql://admin:coffee-admin@coffee.cp82lr4f5r06.us-east-2.rds.amazonaws.com:3306/db?charset=utf8",
        encoding="utf8",
    )
    return db

class CustomBackend(BaseBackend):
    def authenticate(self, request, username=None, password=None, **kwargs):
        User.objects.all().delete()
        login = check_login(username, password)
        
        if login:
            try:
                user = User.objects.get(username=username)
            except User.DoesNotExist:
                user = User(username=username)
                user.is_superuser = True
                user.save()
            return user
        return None
    
    def get_user(self, user_id):
        try:
            return User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return None