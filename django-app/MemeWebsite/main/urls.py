from django.contrib import admin
from django.urls import path, include
from . import views

urlpatterns = [
    path('get_mosaic', views.get_mosaic, name='get_mosaic'),
    path('', views.main),
]
