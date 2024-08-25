var currentImputePage = 1;
var currentAnomalyPage = 1;
var myChart2 = echarts.init(document.getElementById('missing_rate_chart'));
var myChart3 = echarts.init(document.getElementById('anomaly_rate_chart'));

$(function (){
    // 获取补全模型列表
    ImputeModelSelect();
    // 绑定点击 PredictBatchSize 事件
    PredictModelSelect();
    //获取预测模型列表
    PredictWindowSize();
    // Task
    Task();
    // 保存任务配置
    TaskSetSave();
    // 初始化缺失率饼图
    initMissingRateChart();
    // 初始化异常率饼图
    initAnomalyRateChart();
    // 获取表格数据
    ShowTaskResults();
    // 点击 details 按钮
    bindBtnDetails();
    // 保存 details 设置按钮
    bindBtnSaveDetails();
    // 更新异常饼图
    setInterval(updateAnomalyRateChart, 1000);
    // 点击按钮变色
    clickbtncolor();

})

    function ImputeModelSelect() {
        console.log("Initializing ImputeModelSelect");

        var data = {
            models: ["OT", "KNN","Global best"]
        };

        var selectList = $('#ImputeModelSelect');
        selectList.empty(); // 确保列表为空

        $.each(data.models, function(i, model) {
            console.log("Adding model to ImputeModelSelect:", model);
            var listItem = $('<li><label><input type="radio" name="imputeModels" value="' + model + '">' + model + '</label></li>');
            selectList.append(listItem);
        });

        // 使用事件委托绑定 click 事件
        selectList.on('click', 'li', function() {
            var selectedModel = $(this).find('input').val();
            console.log("Impute model selected:", selectedModel);
            $('#ImputeModel').text(selectedModel);
        });
    }

    function PredictModelSelect() {
        console.log("Initializing PredictModelSelect");

        var data = {
            models: ["Transformer","NPTS","ARIMA","Holt-Winters","Linear","Global best"]
        };

        var selectList = $('#PredictModelSelect');
        selectList.empty(); // 确保列表为空

        $.each(data.models, function(i, model) {
            console.log("Adding model to PredictModelSelect:", model);
            var listItem = $('<li><label><input type="radio" name="predictModels" value="' + model + '">' + model + '</label></li>');
            selectList.append(listItem);
        });

        // 使用事件委托绑定 click 事件
        selectList.on('click', 'li', function() {
            var selectedModel = $(this).find('input').val();
            console.log("Predict model selected:", selectedModel);
            $('#PredictModel').text(selectedModel);
        });
    }

    function PredictWindowSize() {
        $('#PredictWindowSize').on('click', 'a', function() {
            $('#BtnPredictWindowSize').html($(this).text() + ' <span class="caret"></span>');
        });
    }

function Task() {
    let intervalId;
    let socket = new WebSocket("ws://localhost:8000/ws/task/")

    socket.onopen = function (e){
        console.log("Connection open");
    };

    document.getElementById('taskToggle').addEventListener('change', (event) => {
      const chartContainerElement = document.getElementById('chart-container');
      if (event.target.checked) {
        document.getElementById('click-trigger').className = 'glyphicon glyphicon-record processing-color';
        socket.send(JSON.stringify({"type": "task.start"}));
    } else {
        socket.send(JSON.stringify({"type": "task.stop"}));
        document.getElementById('click-trigger').className = 'glyphicon glyphicon-record default-color';
        document.getElementById('predict').className = 'glyphicon glyphicon-record default-color';
        document.getElementById('finished').className = 'glyphicon glyphicon-record default-color';
        document.getElementById("Status").textContent = "Stopped"
        clearInterval(intervalId);
      }
    });


    socket.onmessage = function(event){

        console.log(`return message:${event.data}`);
        let data = JSON.parse(event.data);

        // status
        document.getElementById("Status").textContent = data.status;

        if (data.status === "progressing") {
            document.getElementById('predict').className = 'glyphicon glyphicon-record processing-color';
        } else if (data.status === "finished") {
            document.getElementById('finished').className = 'glyphicon glyphicon-record processing-color';
            clearInterval(intervalId);
        }

        let start_time = new Date(data.start_time * 1000);

        if (intervalId) {
            clearInterval(intervalId);
        }
        intervalId = setInterval(() => {
            updateTaskTime(start_time);
        }, 1000);

         if (data.count_nan !== 0 && data.count_not_nan !== 0) {
            updateMissingRateChart(data.count_nan, data.count_not_nan);
        }


    }

    function updateTaskTime(StartTime) {
    let currentTime = new Date();
    if (document.getElementById("Status").textContent !== "finished") {
        let Time = parseInt((currentTime - StartTime) / 1000);
        let formattedImputeTime = formatTime(Time);
        document.getElementById("TaskTime").textContent = formattedImputeTime;
    }
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
        clearInterval(intervalId);
    };

    socket.onerror = function(error) {
        console.log(`[error] ${error.message}`);
    };
}

function TaskSetSave() {
  document.getElementById('TaskSaveToggle').addEventListener('click', (event) => {
    const fetchParams = () => ({
      ImputeModel: $('#ImputeModel').text(),
      PredictModel: $('#PredictModel').text(),
      PredictWindowSize: $('#BtnPredictWindowSize').text().trim()
    });

    const params = fetchParams();

    $.ajax({
      type: "POST",
      url: "/task/save/",
      contentType: "application/json",
      dataType: "JSON",
      data: JSON.stringify(params),
      success: function (data) {
          console.log(JSON.stringify(params));
          console.log("Success: ", data);
      },
      error: function (error) {
          console.log(error);
      }
    });
  });
}



function initMissingRateChart(){
    option_2 = {
  title: {
    text: 'Missing Rate Chart',
    left: 'center',
    top: '18%'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
       top: '5%',
  },
  series: [
    {
      name: 'Missing Rate',
      type: 'pie',
      radius: '50%',
      data: [
          { value: 0, name: 'NaN Values' },
          { value: 0, name: 'Not-NaN Values' }
        ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      },
    }
  ]
};
    myChart2.setOption(option_2);
}

function updateMissingRateChart(newCount_nan, newCount_not_nan) {
    myChart2.setOption({
      series: [{
        data: [
          { value: newCount_nan, name: 'NaN Values' },
          { value: newCount_not_nan, name: 'Not-NaN Values' }
        ],
        type: 'pie'
      }]
    });
  }

function initAnomalyRateChart(){
    option_3 = {
  title: {
    text: 'Anomaly Rate Chart',
    left: 'center',
    top: '18%'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
       top: '0.5%',
  },
  series: [
    {
      name: 'Anomaly Rate',
      type: 'pie',
      radius: '50%',
      data: [
         { value: 0, name: 'Large concurrency' },
         { value: 0, name: 'Out of memory' },
         { value: 0, name: 'Lock race' },
         { value: 0, name: 'Network delay' },
         { value: 0, name: 'Index failure' },
         { value: 0, name: 'Complex query' },
          { value: 0, name: 'Others' }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      },
    }
  ]
};
    myChart3.setOption(option_3);
}

function updateAnomalyRateChart() {
    $.ajax({
        url: "/get_anomaly_data/",
        type: "GET",
        dataType: "json",
        success: function (res) {
            myChart3.setOption({
                series: [
                    {
                        data: res
                    }
                ]
            });
        },
        error: function (xhr, status, error) {
            console.error('AJAX request failed:', status, error);
        }
    });
}



function fetchImputeData(page = currentImputePage) {
    currentImputePage = page;  // 更新当前页码
    $.ajax({
        url: '/load_impute_results/',
        method: 'GET',
        data: {'page': page},
        success: function(data) {
            $('#impute-results-table').html(data.html);  // 更新表格内容
        },
        error: function(xhr, ajaxOptions, thrownError) {
            console.log('Error fetching impute data:', thrownError);
        }
    });
}

function fetchAnomalyData(page = currentAnomalyPage) {
    currentAnomalyPage = page;  // 更新当前页码
    $.ajax({
        url: '/load_anomaly_results/',
        method: 'GET',
        data: {'page': page},
        success: function(data) {
            $('#Anomaly-results-table').html(data.html);  // 更新表格内容
        },
        error: function(xhr, ajaxOptions, thrownError) {
            console.log('Error fetching anomaly data:', thrownError);
        }
    });
}

function ShowTaskResults(){
     // 初始加载数据
    fetchImputeData();
    fetchAnomalyData();

    // 每秒钟更新一次表格
    setInterval(fetchImputeData, 1000);
    setInterval(fetchAnomalyData, 1000);

    // 处理分页点击事件
    $(document).on('click', '.pagination1 a', function (e) {
        e.preventDefault();
        var page = $(this).data('page');
        fetchImputeData(page);
    });

     $(document).on('click', '.pagination2 a', function(e) {
        e.preventDefault();
        var page = $(this).data('page');
        fetchAnomalyData(page);
    });

    // 处理表单提交事件
    $('#impute-results-table').on('submit', 'form', function(e) {
        e.preventDefault();
        var page = $("input[name='page']", this).val();
         fetchImputeData(page);
    });

    $('#Anomaly-results-table').on('submit', 'form', function(e) {
        e.preventDefault();
        var page = $("input[name='page']", this).val();
        fetchAnomalyData(page);
    });
}

function bindBtnDetails() {
    console.log("Binding button details.");
    $('#Anomaly-results-table').on('click', '.btn-edit', function() {
        console.log("Button clicked.");
        //清空对话框中的数据
        $('#formAdd')[0].reset();
        var currentId = $(this).attr('uid');
        EDIT_ID = currentId;
        $("#myModal").modal('show');
        $.ajax({
            url: "/get/analysis/",
            type: 'GET',
            data: {uid: currentId},
            dataType: "JSON",
            success: function (data) {
                if(data.status == 'ERROR'){
                    alert('Error: ' + data.error);
                } else {
                    $("#analysisInput").val(data.analysis);
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                //服务器连接失败时的处理
                alert('Failed to connect to server: ' + textStatus);
            }
        });
    })
}

function bindBtnSaveDetails(){
    $("#btnSave").click(function () {
    $.ajax({
        url: "/save/analysis/",
        type: 'POST',
        data: {
            uid: EDIT_ID,
            analysis: $("#analysisInput").val()
        },
        dataType: "JSON",
        success: function (data) {
            if(data.status == 'OK'){
                alert('Saved successfully');
                $("#myModal").modal('hide'); // 关闭模态框
                // 添加代码来更新页面的其他元素，例如列表或者数据表格，来显示新的数据
            } else {
                alert('Failed to save: ' + data.error);
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert('Failed to connect to server: ' + textStatus);
        }
    });
})
}

function clickbtncolor(){
    $(".btn-group-xs .btn").click(function(){
        // 先重置所有按钮的颜色
        $(".btn-group-xs .btn").removeClass("btn-selected");
        $(".btn-group-xs .btn").addClass("btn-default");

        // 为选中的按钮添加新样式
        $(this).removeClass("btn-default");
        $(this).addClass("btn-selected");
    });
}


