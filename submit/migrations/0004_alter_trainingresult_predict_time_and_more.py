# Generated by Django 4.1 on 2024-08-10 15:36

from django.db import migrations, models


class Migration(migrations.Migration):
    dependencies = [
        ("submit", "0003_trainingresult"),
    ]

    operations = [
        migrations.AlterField(
            model_name="trainingresult",
            name="predict_time",
            field=models.FloatField(),
        ),
        migrations.AlterField(
            model_name="trainingresult",
            name="train_time",
            field=models.FloatField(),
        ),
    ]
