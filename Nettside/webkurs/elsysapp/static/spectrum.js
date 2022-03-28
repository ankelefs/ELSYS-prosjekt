function draw() {
    /* Accepting and seperating comma seperated values */
    var n = document.getElementById("num").value;
    var values = n.split(',');
    
    var canvas = document.getElementById('myCanvas');
    var ctx = canvas.getContext('2d');

    var width = 5; //bar width
    var X = 50; // first bar position 
    var base = 200;
    
    for (var i =0; i<values.length; i++) {
        ctx.fillStyle = '#008080'; 
        var h = values[i];
        ctx.fillRect(X,canvas.height - h,width,h);
        
        X +=  width+15;
        /* text to display Bar number */
        
    }
        

}
function reset(){
     var canvas = document.getElementById('myCanvas');
      var ctx = canvas.getContext('2d');
       ctx.clearRect(0, 0, canvas.width, canvas.height);
}