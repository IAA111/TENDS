import asyncio
import json
import time
from asgiref.sync import sync_to_async
from submit.models import Task
from channels.consumer import AsyncConsumer
import csv
import pandas as pd
from . import models
from . import views

class TaskChatConsumer(AsyncConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = None
        self.start_time = None
        self.status = None
        self.impute_model = None
        self.predict_model = None
        self.predict_window_size = None

    async def websocket_connect(self, event):
        print("connected", event)
        await self.send({
            "type": "websocket.accept"
        })

    async def websocket_receive(self, event):
        print("received", event)
        text_data = json.loads(event['text'])
        message = text_data['type']

        if message == "task.start":
            if self.task:
                self.task.cancel()
            self.task = asyncio.ensure_future(self.start_task())

        elif message == "task.stop":
            if self.task:
                self.task.cancel()
                self.task = None

    async def websocket_disconnect(self, event):
        print("disconnected", event)

    async def start_task(self):
        # 获取参数
        task_parameters = await sync_to_async(Task.objects.last, thread_sensitive=True)()
        self.impute_model = task_parameters.impute_model
        self.predict_model = task_parameters.predict_model
        self.predict_window_size = task_parameters.predict_window_size

        self.start_time = time.time()
        self.status = "progressing"
        await self.send_status()

        await self.impute()
        await self.predict()

        self.status = "finished"
        await self.send_status()

    async def send_status(self):
        await self.send({
            "type": "websocket.send",
            "text": json.dumps({
                "status": self.status,
                "start_time": self.start_time,
            })
        })

    async def impute(self):

        print("开始执行补全")

        '''  

             具体补全过程

        '''
        await asyncio.sleep(10)


    async def predict(self):
        print("开始执行预测")

        await asyncio.sleep(5)