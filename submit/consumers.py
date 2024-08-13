import asyncio
import json
import time
import asyncio
from asgiref.sync import sync_to_async
from submit.models import Task,PreData,ImputeResult,AnomalyResult
from channels.consumer import AsyncConsumer
from geomloss import SamplesLoss
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
import numpy as np
from submit.imp_utils import *
import csv
import pandas as pd
import torch
from . import models
from . import views
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima

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
            niter = 30
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

        f = "media/dataset/onehr.csv"
        df = pd.read_csv(f, parse_dates=[0], header=None, index_col=0, date_format='%m/%d/%Y', nrows=500, usecols=range(7))
        # df.values 将 DataFrame 转换为 NumPy 数组
        X = df.values[::].astype('float')  # 得到缺失数组

        time_index = df.index.values
        # 时间数组 formatted_time_index
        formatted_time_index = df.index.strftime('%Y-%m-%d').values

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

        # predict
        # 定义预测长度
        train_len = int(len(sk_imp) * 0.7)
        prediction_len = len(sk_imp) - train_len

        df_sk_imp = pd.DataFrame(sk_imp)
        data = df_sk_imp[:train_len]
        # 创建一个空的DataFrame来存储预测结果
        # 使用预测结果的索引
        predictions = pd.DataFrame(index=pd.RangeIndex(start=len(data), stop=len(data) + prediction_len))

        # 循环遍历DataFrame的每一列
        for column in data.columns:
            # 获取当前列的数据
            history_data = data[column]

            predictor = ETSModel(history_data.squeeze(), error="add", trend="additive", seasonal="add",
                                 seasonal_periods=4).fit()
            pred_res = predictor.forecast(prediction_len)

            # 将预测结果添加到predictions DataFrame结构中
            predictions[column] = pred_res

        predict = predictions.values
        print(predict)
        print(train_len, prediction_len, len(sk_imp))  # 1052 451 1503

        # 异常检测
        predicted_mask = []
        for i in range(0, prediction_len):
            row_mask = []
            for j in range(len(sk_imp[i])):
                if sk_imp[i + train_len][j] != 0:
                    diff_percentage = abs((sk_imp[i + train_len][j] - predict[i][j]) / sk_imp[i + train_len][j])
                    if diff_percentage > 1.2:  # 异常系数
                        row_mask.append(True)
                    else:
                        row_mask.append(False)
                else:
                    row_mask.append(False)
            predicted_mask.append(row_mask)
        predicted_mask = np.array(predicted_mask)



        save_predata = sync_to_async(PreData.save)

        # 存储补全后数据至PreData  前70%未预测部分  用于chart数据提取
        for i in range(train_len):
            data_str = ','.join(map(lambda x: '{:.2f}'.format(x), sk_imp[i]))
            mask_str = ','.join(map(str, mask[i]))
            predata = PreData(
                time=formatted_time_index[i],
                index=i,
                data=data_str,
                mask=mask_str,
                predicted_data='0',
                predicted_mask='0',
            )
            await save_predata(predata)

        # 存储预测后数据至PreData  后30%预测部分
        for i in range(prediction_len):
            data_str = ','.join(map(lambda x: '{:.2f}'.format(x), sk_imp[i + train_len]))
            mask_str = ','.join(map(str, mask[i + train_len]))
            predicted_data_str = ','.join(map(lambda x: '{:.2f}'.format(x), predict[i]))
            predicted_mask_str = ','.join(map(lambda x: str(x), predicted_mask[i]))
            predata = PreData(
                time=formatted_time_index[i + train_len],
                index=i + train_len,
                data=data_str,
                mask=mask_str,
                predicted_data=predicted_data_str,
                predicted_mask=predicted_mask_str,
            )
            await save_predata(predata)

        # 存储缺失点信息 impute_result
        save_impute_result = sync_to_async(ImputeResult.save)
        count = 1
        for i, row in enumerate(sk_imp):
            for j, value in enumerate(row):
                if mask[i, j]:
                    # 仅保存缺失值的补全结果
                    impute_result = ImputeResult(
                        time = formatted_time_index[i],
                        index=i,
                        variable=j + 1,  # 列号从 1 开始
                        Imputed_value=value,
                        count = count
                    )
                    # 使用 await 调用异步保存方法
                    await save_impute_result(impute_result)
                    count += 1


        # 存储异常信息
        save_anomaly_result = sync_to_async(AnomalyResult.save)
        count = 1
        for i in range(prediction_len):
            for j in range(len(predicted_mask[i])):
                if predicted_mask[i][j]:
                    anomaly_result = AnomalyResult(
                        time=formatted_time_index[i + train_len],
                        index=i + train_len,
                        count=count,
                        variable=j + 1,  # 列号从 1 开始
                        true_value=sk_imp[i + train_len][j],
                        predict_value=predict[i][j],
                        analysis='',  # 可以在这里填写一些分析备注或保持默认
                    )
                    await save_anomaly_result(anomaly_result)
                    count += 1


    async def predict(self):
        print("开始执行预测")

        await asyncio.sleep(15)


class TrainChatConsumer(AsyncConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_task = None
        self.start_time = None
        self.impute_total_model = None
        self.impute_model_count = None
        self.predict_total_model = None
        self.predict_model_count = None

    # WebSocket连接成功
    async def websocket_connect(self, event):
        print("connected", event)
        await self.send({
            "type": "websocket.accept"
        })

    # 当前端发送消息到服务器时
    async def websocket_receive(self, event):
        print("received", event)
        text_data = json.loads(event['text'])
        message = text_data['type']

        if message == "training.start":
            if self.training_task:
                self.training_task.cancel()
            self.training_task = asyncio.ensure_future(self.start_training())

        elif message == "training.stop":
            if self.training_task:
                self.training_task.cancel()
                self.training_task = None

    # WebSocket连接断开
    async def websocket_disconnect(self, event):
        print("disconnected", event)

    async def start_training(self):
        print("start train_all_models")
        self.start_time = time.time()
        model_parameters = await sync_to_async(models.TrainParameters.objects.last, thread_sensitive=True)()
        self.impute_model = model_parameters.impute_model.split(',')
        self.predict_model = model_parameters.predict_model.split(',')
        self.impute_model_count = 0
        self.predict_model_count = 0
        self.impute_total_model = len(self.impute_model)
        self.predict_total_model = len(self.predict_model)
        await self.send_status()
        # 获取训练参数
        '''
        model_parameters = await sync_to_async(TrainParameters.objects.last, thread_sensitive=True)()
        self.impute_model = model_parameters.impute_model.split(',')
        self.predict_model = model_parameters.predict_model.split(',')
        self.train_data_size = model_parameters.train_data_size
        self.predict_window_size = model_parameters.predict_window_size
        self.imputation_size = model_parameters.imputation_size
        self.impute_model_count = 0
        self.predict_model_count = 0
        self.impute_total_model = len(self.impute_model)
        self.predict_total_model = len(self.predict_model)

        if model_parameters.dataset:
            self.dataset = model_parameters.dataset.open('r')  # 读取 dataset 文件

        await self.impute()
        self.dataset.close()  # 关闭 dataset 文件
        await self.train_all_models()
        '''

    async def send_status(self):
        await self.send({
            "type": "websocket.send",
            "text": json.dumps({
                "start_time": self.start_time,
                "impute_total_model": self.impute_total_model,
                "impute_model_count": self.impute_model_count,
                "predict_total_model": self.predict_total_model,
                "predict_model_count": self.predict_model_count,
            })
        })


