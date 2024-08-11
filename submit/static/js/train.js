$(function (){
    var selectedClass = 'myClass';
    // 预测模型下拉内容
    PredictModelSelect();
    // 补全模型下拉内容
    ImputeModelSelect();
    // 绑定点击 BtnTrainBatchSize 事件
    TrainDataSize();
    // 绑定点击 BtnPredictBatch 事件
    PredictWindowSize();
    // 绑定点击 BtnTrainSetSave 按钮事件
    binBtnTrainSetSave();
    // 绑定点击 PredictBatch 事件
    ImputationSize();
    // 绑定点击 StartTrainToggle 事件
    StartTrain();
    // 训练结果表单分页
    ShowTrainResults();
    // 显示选中文件名
    showfilename();
    // 模态框保存
    binBtnSave();
    // 处理分页点击事件

})

function TrainDataSize(){
    $('#TrainDataSize a').click(function(){
        $('#BtnTrainDataSize').html($(this).text() + ' <span class="caret"></span>');
    });
}

function PredictWindowSize(){
    $('#PredictWindowSize a').click(function(){
        $('#BtnPredictWindowSize').html($(this).text() + ' <span class="caret"></span>');
    });
}

function ImputationSize(){
    $('#ImputationSize a').click(function(){
        $('#BtnImputationSize').html($(this).text() + ' <span class="caret"></span>');
    });
}

function PredictModelSelect(){
    var data = {
        models: ["Model 1", "Model 2", "Model 3", "Model 4"]
    };

    var selectList = $('#TrainPredictModel');

    $.each(data.models, function(i, model) {
        selectList.append('<li><label><input type="checkbox" value="' + model + '">' + model + '</label></li>');
    });

    $('.dropdown-menu').on('click', function(e) {
        if($(e.target).is('input[type="checkbox"]')) {
            e.stopImmediatePropagation();
        }
    });
}

function ImputeModelSelect(){

    $(document).ready(function() {
    var data = {
        models: ["Model 1", "Model 2", "Model 3", "Model 4"]
    };

    var selectList = $('#TrainImputeModel');

    $.each(data.models, function(i, model) {
        selectList.append('<li><label><input type="checkbox" value="' + model + '">' + model + '</label></li>');
    });

    $('.dropdown-menu').on('click', function(e) {
        if($(e.target).is('input[type="checkbox"]')) {
            e.stopImmediatePropagation();
        }
    });
});
}

function binBtnTrainSetSave() {
    $('#BtnTrainSetSave').click(function () {
        var form_data = new FormData();
        form_data.append("dataset", $('#upload')[0].files[0]);

        const fetchParams = () => ({
            impute_model: $('#TrainImputeModel input[type="checkbox"]:checked').map(function () {
                return this.value;
            }).get(),
            predict_model: $('#TrainPredictModel input[type="checkbox"]:checked').map(function () {
                return this.value;
            }).get(),
            train_data_size: $("#BtnTrainDataSize").text().trim(),
            predict_window_size: $("#BtnPredictWindowSize").text().trim(),
            imputation_size:$("#BtnImputationSize").text().trim(),
        });

        const params = fetchParams();

        for ( var key in params ) {
             form_data.append(key, params[key]);
         }

        $.ajax({
            type: "POST",
            url: "/train/save/",
            data: form_data,
            processData: false,
            contentType: false,
            success: function (data) {
                console.log("Success: ", data);


                //清空对话框中的数据
            $('#formAdd')[0].reset();

            $.ajax(
                {
                    url:"/order/detail/",
                    type:'get',
                    dataType:"JSON",
                    success: function (res){
                        if (res.status){
                            console.log(res.data)
                            // 将数据赋值到对话框的标签中
                            $.each(res.data,function(name,value){
                                $('#id_'+name).val(value);
                            })

                            $('#myModal').modal('show')
                        }else {
                            alert(res.error)
                        }
                    }

                }
            )
            },
            error: function (jqXHR, textStatus, errorThrown) {
             console.log(params);
             console.log("jqXHR: ", jqXHR);
             console.log("textStatus: ", textStatus);
             console.log("Error details: ", errorThrown);
             console.log(JSON.stringify(params));
   }})
        })
}



function StartTrain(){
    let intervalId;
    let socket = new WebSocket("ws://localhost:8000/ws/train/")

    socket.onopen = function (e){
         console.log("Connection open");
    };

    document.getElementById('StartTrainToggle').addEventListener('change', (event) => {
      if (event.target.checked) {
        socket.send(JSON.stringify({"type": "training.start"}));
      } else {
        socket.send(JSON.stringify({"type": "training.stop"}));
        intervalId && clearInterval(intervalId);
        start_time = undefined;
      }
    });

    socket.onmessage = function(event){
        console.log(`return message:${event.data}`);

        let data = JSON.parse(event.data)
        console.log(`Start time received from server: ${data.start_time}`);

        let start_time = new Date(data.start_time * 1000);

        if (intervalId) {
            clearInterval(intervalId);
        }
        intervalId = setInterval(() => {
            updateTaskTime(start_time);
        }, 1000);

        document.getElementById("PreModelCount").textContent = data.predict_model_count + "/" + data.predict_total_model;
        document.getElementById("ImpModelCount").textContent = data.impute_model_count + "/" + data.impute_total_model;

    }

    function updateTaskTime(StartTime) {
        let currentTime = new Date();
        let Time = parseInt((currentTime - StartTime) / 1000);
        let formattedImputeTime = formatTime(Time);
        document.getElementById("imputeTaskTime").textContent = formattedImputeTime;
    }

// 将获得的秒数转换为 HH:MM:SS 格式
    function formatTime(seconds) {
        let hours = parseInt(seconds / 3600);
        let minutes = parseInt((seconds % 3600) / 60);
        let remainingSeconds = seconds % 60;

        return `${hours}:${minutes}:${remainingSeconds}`;
    }

    socket.onclose = function(event) {
        if (event.wasClean) {
            console.log(`Connection closed properly：${event.code},${event.reason}`);
        } else {
            console.log('disconnect');
        }
        intervalId && clearInterval(intervalId);
        start_time = undefined;
    };

    socket.onerror = function(error) {
        console.log(`[error] ${error.message}`);
    };

}

function ShowTrainResults() {
    $(document).on('click', '.pagination a', function (e) {
        e.preventDefault();  // 阻止默认行为

        var page = $(this).data('page');

        $.ajax({
            url: '/load_train_results/',
            data: {'page': page},
            method: 'GET',
            success: function (data) {
                $('#train-results-table').html(data.html);
            },
            error: function (xhr, ajaxOptions, thrownError) {
                console.log(thrownError);
            }
        });
    });
    $(document).on('submit', 'form', function (event) {
        event.preventDefault();
        var page = $("input[name='page']", this).val();

        $.ajax({
            url: '/load_train_results/',
            type: 'GET',
            data: {'page': page},
            success: function (data) {
                $('#train-results-table').html(data.html);
            }
        });
    });
}

function showfilename(){
    $('#upload').change(function() {
        var filename = $(this).val().split('\\').pop();
        $('.custom-file-upload').text(filename);
    });
}

function redirectToURL() {
    window.location.href = "http://localhost:8000/online/";
}

function binBtnSave(){
     $('#btnSave').click(function() {
          console.log('Button clicked!');
        console.log('Hiding modal...');
        $('#myModal').modal('hide');
    });
}