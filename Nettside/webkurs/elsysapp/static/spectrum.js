function loadFile() {
    var input, file, fr;

    if (typeof window.FileReader !== 'function') {
        bodyAppend("p", "The file API isn't supported on this browser yet.");
        return;
    }

    input = document.getElementById('fileinput');
    
    if (!input) {
        bodyAppend("p", "Um, couldn't find the fileinput element.");
    }
    else if (!input.files) {
        bodyAppend("p", "This browser doesn't seem to support the `files` property of file inputs.");
    }
    else if (!input.files[0]) {
        bodyAppend("p", "Please select a file before clicking 'Load'");
    }
    else {
        file = input.files[0];
        fr = new FileReader();
        fr.readAsBinaryString(file);
        fr.onload = receivedBinary;
        
    }

    function receivedBinary() {
        showResult(fr);
    }
       
}

let markup = [];


function showResult(fr) {
    var result, n, aByte, byteStr, i, canvas, width, X, base, h;
    result = fr.result;
    for (n = 0; n < result.length; ++n) {
        aByte = result.charCodeAt(n);
        byteStr = aByte.toString(10);
        if (byteStr.length < 2) {
            byteStr = "0" + byteStr;
        }
        markup.push(byteStr);
    }
    canvas = document.getElementById('myCanvas');
    ctx = canvas.getContext('2d');

    width = 2; //bar width
    X = 0; // first bar position 
    base = 100;
    
    for (i = 0; i < markup.length; i++) {
        ctx.fillStyle = '#008080'; 
        var h = markup[i];
        ctx.fillRect(X,canvas.height - h,width,h);
        
        X +=  width+15;      
    }
}


function draw() {
    var i, canvas, ctx, width, X, base, h;
    /* Accepting and seperating comma seperated values */

    canvas = document.getElementById('myCanvas');
    ctx = canvas.getContext('2d');

    width = 5; //bar width
    X = 50; // first bar position 
    base = 200;
    
    for (var i =0; i<markup.length; i++) {
        ctx.fillStyle = '#008080'; 
        var h = markup[i];
        ctx.fillRect(X,canvas.height - h,width,h);
        
        X +=  width+15;      
    }
}

function reset(){
    var canvas = document.getElementById('myCanvas');
    var ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    draw();
}

