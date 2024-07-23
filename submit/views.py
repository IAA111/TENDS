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

@csrf_exempt
def task_save(request):
    if request.method == "POST":
        data = json.loads(request.body)

        PredictWindowSizestr = data.get('PredictWindowSize')
        PredictWindowSize = float(PredictWindowSizestr.strip('%')) / 100

        obj = models.Task(
            impute_model=data.get('ImputeModel'),
            predict_model=data.get('PredictModel'),
            predict_window_size=PredictWindowSize,
        )
        obj.save()
        print(data)
        return JsonResponse({"message": "Parameters were saved successfully."})
    else:
        return JsonResponse({"error": "error."})

def get_chart_data(request):
    PREDICTION_START_POINT = 2000  # 开始预测位置的索引
    data = models.PreData.objects.all().order_by('index').values('index', 'data', 'predicted_data', 'mask',
                                                                 'predicted_mask', 'time')
    print(data)

    '''"   time": data["time"].strftime('%Y-%m-%d %H:%M:%S'),   '''
    formatted_data = [
        {
         "index": data["index"],
         "figures": [float(i) for i in data["data"].split(",")],
         "predicted_figures": [float(i) if data["index"] >= PREDICTION_START_POINT else None for i in
                               data["predicted_data"].split(",")],
         "highlighted_figures": [float(d) if m == 'True' else None for d, m in
                                 zip(data["data"].split(","), data["mask"].split(","))],
         "highlighted_predicted_figures": [float(d) if m == 'True' else None for d, m in
                                           zip(data["predicted_data"].split(","), data["predicted_mask"].split(","))]
         } for data in data
    ]

    return JsonResponse(formatted_data, safe=False)
