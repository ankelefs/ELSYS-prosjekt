$(function () {
          
    var $sensorChart = $("#sensor-chart");
    $.ajax({
      url: $sensorChart.data("url"),
      success: function (data) {

        var ctx = $sensorChart[0].getContext("2d");
        console.log(data);

        new Chart(ctx, {
          type: 'bar',
          data: {
            labels: data.labels,
            datasets: [{
              label: 'Sensoraktivitet',
              backgroundColor: 'black',
              data: data.data
            }]          
          },
          options: {
            responsive: true,
            legend: {
              position: 'top',
            },
            title: {
              display: true,
              text: 'Sensoraktivitet'
            }
          }
        });

      }
    });

  });