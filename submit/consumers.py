import asyncio
import json
import time
import asyncio
from asgiref.sync import sync_to_async
from submit.models import Task,PreData
from channels.consumer import AsyncConsumer
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler
import numpy as np
from submit.imp_utils import *
import csv
import pandas as pd
import torch
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
        self.count_nan = 0
        self.count_not_nan = 0


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
        # get task parameters
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
        try:
            await self.send({
                "type": "websocket.send",
                "text": json.dumps({
                    "status": self.status,
                    "start_time": self.start_time,
                    "count_nan": int(self.count_nan),
                    "count_not_nan": int(self.count_not_nan),
                })
            })
        except Exception as e:
            print(f"Error sending status: {e}")

    async def impute(self):
        print("开始执行补全")
        await sync_to_async(PreData.objects.all().delete)()

        def OTimputer(X, eps, X_true):
            batchsize = 128
            niter = 3000
            OT_lr = 1e-2
            noise = 0.1
            n_pairs = 1
            opt = torch.optim.RMSprop

            sk = SamplesLoss("sinkhorn", p=2, blur=eps, scaling=0.9, backend="tensorized")

            X = X.clone()
            n, d = X.shape

            if batchsize > n // 2:
                e = int(np.log2(n // 2))
                batchsize = 2 ** e

            mask = torch.isnan(X).double()

            # initialization
            imps = (noise * torch.randn(mask.shape).double() + nanmean(X, 0))[mask.bool()]
            imps.requires_grad = True

            optimizer = opt([imps], lr=OT_lr)

            maes = np.zeros(niter)
            rmses = np.zeros(niter)

            for i in range(niter):
                # initialize
                X_filled = X.detach().clone()
                X_filled[mask.bool()] = imps
                loss = 0
                for _ in range(n_pairs):
                    idx1 = np.random.choice(n, batchsize, replace=False)
                    idx2 = np.random.choice(n, batchsize, replace=False)

                    X1 = X_filled[idx1]
                    X2 = X_filled[idx2]

                    loss += sk(X1, X2)  # Sinkhorn Distance

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                maes[i] = MAE(X_filled, X_true, mask).item()
                rmses[i] = RMSE(X_filled, X_true, mask).item()

            X_filled = X.detach().clone()
            X_filled[mask.bool()] = imps

            return X_filled

        f = "/Users/sherryd/Desktop/asf1_0.1miss.csv"

        start_time = time.time()

        df = pd.read_csv(f, header=0)
        # df.values 将 DataFrame 转换为 NumPy 数组
        X = df.values[::].astype('float')  # 得到缺失数组

        # 得到 mask nan 对应 True
        mask = np.isnan(X)

        # 统计 missing rate
        self.count_nan = np.sum(mask)
        self.count_not_nan = np.sum(~mask)

        # 同一列上下非nan邻居值的均值补全
        X = fillna_mean_of_neighbors(X)

        # 进行标准化
        scaler = StandardScaler()
        scaler.fit(X)
        ground_truth = scaler.transform(X)
        X_true = torch.tensor(ground_truth)

        # 补全部分
        mask = torch.from_numpy(mask)
        X_nas = X_true.clone()
        X_nas[mask.bool()] = np.nan  # 将 X_nas 中 mask 为 True 的位置设置为 NaN  缺失数据

        M = mask.sum(1) > 0
        nimp = M.sum().item()  # 缺失值数量

        quantile = .5
        quantile_multiplier = 0.05
        epsilon = pick_epsilon(X_nas, quantile, quantile_multiplier)

        sk_imp = OTimputer(X_nas.clone(), epsilon, X_true)
        sk_imp = sk_imp.detach()

        sk_imp = scaler.inverse_transform(sk_imp)

        mask = mask.numpy()

        save_predata = sync_to_async(PreData.save)

        for i, row in enumerate(sk_imp):
            data_str = ', '.join(map(str, row))
            mask_str = ', '.join(map(str, mask[i]))
            predata = PreData(
                index=i,
                data=data_str,
                mask=mask_str,
                predicted_data='0',
                predicted_mask='0',

            )
            # 使用await调用异步保存方法
            await save_predata(predata)
            



    async def predict(self):
        print("开始执行预测")

        await asyncio.sleep(5)