function draw_board(board, stones, canvas){
    var ctx = canvas.getContext('2d');
    ctx.fillStyle = '#8080FF';
    ctx.fillRect(0,0,canvas.width, canvas.height);
    
    draw_grid(ctx, board);

    draw_stones(ctx, board, stones);
}

function draw_stones(ctx, board, stones){
    stones.forEach(function(stone, index){
        draw_stone(ctx, board, stone.x, stone.y, index, index==stones.length-1);
    });
}

function draw_line(context, x1, y1, x2, y2){
    context.beginPath(); 
    context.moveTo(x1, y1);
    context.lineTo(x2, y2);
    context.strokeStyle="#C0C0C0";
    context.stroke();
}

function draw_grid(ctx, board) {
    var step = board.size / (board.squares+2);
    x0 = 1.5 * step;
    x1 = board.size - 1.5 * step;
    for (i = 0; i < board.squares; i++){
        y = 1.5*step + i*step;
        draw_line(ctx, x0, y, x1, y);
    }
    y0 = 1.5 * step;
    y1 = board.size - 1.5 * step;
    for (i = 0; i < board.squares; i++){
        x = 1.5 * step + i*step;
        draw_line(ctx, x, y0, x, y1);
    }

    ctx.fillStyle = "#FFFFFF";
    ctx.font = "14px verdana, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (i = 0; i < board.squares; i++){    
        ctx.fillText(String.fromCharCode(65+i), (i+1.5)*step, board.size-.5*step);
        ctx.fillText(String.fromCharCode(65+i), (i+1.5)*step, step);
    }
    ctx.textAlign = "right";
    ctx.textBaseline = "center";
    for (i = 0; i < board.squares; i++){    
        ctx.fillText(board.squares-i, step, (i+1.7)*step);
        ctx.fillText(board.squares-i, board.size-0.5*step, (i+1.7)*step);
    }
}

function draw_stone(ctx, board, x, y, index, marker=false){

    color = index % 2;
    if ( color == 1 ){
        ctx.fillStyle="#FFFFFF";
    } else {
        ctx.fillStyle="#000000";
    }
    
    step = board.size / (board.squares+2);
    
    var x = step/2 + x * step; // x coordinate
    var y = 1.5*step + (board.squares-y) * step; // y coordinate
    var radius = step/2 - 1; // Arc radius
    var startAngle = 0; // Starting point on circle
    var endAngle = 2*Math.PI; // End point on circle

    ctx.beginPath();
    ctx.arc(x, y, radius, startAngle, endAngle, true);
    ctx.fill();
    
    if ( color == 0 ){
        ctx.fillStyle="#FFFFFF";
    } else {
        ctx.fillStyle="#000000";
    }
    ctx.font = "14px verdana, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "center";
    ctx.fillText(index+1, x, y+.25*step);

    if (marker){
        ctx.lineWidth=2;
        ctx.strokeStyle="#FF0000";        
        ctx.beginPath();
        ctx.arc(x, y, radius+1, startAngle, endAngle, true);
        ctx.stroke()
    }
}

