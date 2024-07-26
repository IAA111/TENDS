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
    models.PreData.objects.all().delete()
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
    PREDICTION_START_POINT = 3000
    data = models.PreData.objects.all().order_by('index').values()
    formatted_data = []
    for row in data:
        index = row['index']
        figures = [float(i) for i in row['data'].split(',')]

        # 去除mask中的空格并转换为布尔值
        mask_values = [m.strip() == 'True' for m in row['mask'].split(',')]

        highlighted_figures = [float(d) if m else None for d, m in zip(figures, mask_values)]
        print("Mask:", row['mask'])
        print("Mask Split:", row['mask'].split(','))
        print("Processed Mask:", mask_values)
        print("Zipped Figures and Mask:")
        for d, m in zip(figures, mask_values):
            print(f"\tFigure: {d}, Mask Value: {m}")
        print(highlighted_figures)

        for d, m in zip(figures, mask_values):
            print(f"\tFigure: {d}, Mask Value: {m}")

        if index < PREDICTION_START_POINT:
            predicted_figures = [None] * len(figures)
            highlighted_predicted_figures = [None] * len(figures)
        else:
            predicted_data_list = row['predicted_data'].split(',')
            predicted_figures = [float(i) if i else None for i in predicted_data_list]

            # 同样地，处理predicted_mask
            predicted_mask_values = [m.strip() == 'True' for m in row['predicted_mask'].split(',')]
            highlighted_predicted_figures = [float(d) if m else None for d, m in
                                             zip(predicted_figures, predicted_mask_values)]

        formatted_data.append({
            "index": index,
            "figures": figures,
            "predicted_figures": predicted_figures,
            "highlighted_figures": highlighted_figures,
            "highlighted_predicted_figures": highlighted_predicted_figures,
        })

    return JsonResponse(formatted_data, safe=False)

def show_chart(request):
    return render(request, 'chart.html')
