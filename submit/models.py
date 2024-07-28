from django.db import models

# Create your models here.

# task parameter
class Task(models.Model):
    impute_model = models.CharField(max_length=20)
    predict_model = models.CharField(max_length=20)
    predict_window_size = models.FloatField()

class PreData(models.Model):
    index = models.IntegerField()
    data = models.TextField()
    mask = models.TextField()
    predicted_data = models.TextField(default='')
    predicted_mask = models.TextField(default='')
    time = models.DateTimeField(null=True, blank=True)

class ImputeResult(models.Model):
    time = models.CharField(max_length=24,null=True, blank=True)
    index = models.IntegerField()
    variable = models.IntegerField()
    Imputed_value = models.FloatField()

class AnomalyResult(models.Model):
    time = models.CharField(max_length=24,null=True, blank=True)
    index = models.IntegerField()
    variable = models.IntegerField()
    true_value = models.FloatField()
    predict_value = models.FloatField()
    analysis = models.CharField(max_length=255, default='')

class TrainParameters(models.Model):
    impute_model = models.TextField()
    predict_model = models.TextField()
    train_data_size = models.FloatField()
    predict_window_size = models.FloatField()
    imputation_size = models.FloatField()
    dataset = models.FileField(verbose_name='dataset',max_length=128,upload_to='dataset/')



