"""TENDS URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/4.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path,re_path
from submit import views
from django.conf import settings
from django.views.static import serve

urlpatterns = [
    path("offline/",views.offline),
    path("online/",views.online),
    path('task/save/', views.task_save),
    path("get_chart_data/",views.get_chart_data),
    path('load_impute_results/', views.load_impute_results),
    path('load_anomaly_results/', views.load_anomaly_results),
    path('get/analysis/', views.get_analysis),
    path('save/analysis/', views.save_analysis),
    path('get_anomaly_data/', views.get_anomaly_data),
    path('train/save/', views.train_save),
    path("order/detail/",views.order_detail),
    path('load_train_results/', views.load_train_results),


]
