from django.shortcuts import render

# Create your views here.
from django.shortcuts import render, HttpResponse, redirect
from django.http import JsonResponse
from django.template.loader import render_to_string
from django.views.decorators.csrf import csrf_exempt
import json
from django.core import serializers
from submit import models

def offline(request):
    return render(request, 'home.html')

def online(request):
    return render(request, 'predict.html')