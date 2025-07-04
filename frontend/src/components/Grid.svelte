<script lang="ts">
    import { onMount } from 'svelte';
	import { neural_network_response } from './prediction.svelte';
    
    let canvas: HTMLCanvasElement;
    let ctx: CanvasRenderingContext2D;
    let isDrawing: boolean = false;
    let grid: number[][] = Array(28).fill(null).map(() => Array(28).fill(0));
    
    const GRID_SIZE: number = 28;
    const CELL_SIZE: number = 20;
    const CANVAS_SIZE: number = GRID_SIZE * CELL_SIZE;
    
    onMount(() => {
      const context = canvas.getContext('2d');
      if (context) {
        ctx = context;
        drawGrid();
      }
    });
    
    function drawGrid(): void {
      if (!ctx) return;
      
      ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
      
      ctx.strokeStyle = '#FFF';
      ctx.lineWidth = 1;
      
      for (let i = 0; i <= GRID_SIZE; i++) {
        ctx.beginPath();
        ctx.moveTo(i * CELL_SIZE, 0);
        ctx.lineTo(i * CELL_SIZE, CANVAS_SIZE);
        ctx.stroke();
        
        ctx.beginPath();
        ctx.moveTo(0, i * CELL_SIZE);
        ctx.lineTo(CANVAS_SIZE, i * CELL_SIZE);
        ctx.stroke();
      }
      
      for (let row = 0; row < GRID_SIZE; row++) {
        for (let col = 0; col < GRID_SIZE; col++) {
          if (grid[row][col] > 0) {
            const intensity = grid[row][col];
            ctx.fillStyle = `rgba(${intensity}, ${intensity}, ${intensity}, 1)`;
            ctx.fillRect(col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE);
          }
        }
      }
    }
    
    function getGridPosition(event: MouseEvent): { row: number; col: number } {
      const rect = canvas.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      
      const col = Math.floor(x / CELL_SIZE);
      const row = Math.floor(y / CELL_SIZE);
      
      return { row, col };
    }
    
    function startDrawing(event: MouseEvent): void {
      isDrawing = true;
      draw(event);
    }
    
    function draw(event: MouseEvent): void {
      if (!isDrawing) return;
      
      const { row, col } = getGridPosition(event);
      
      if (row >= 0 && row < GRID_SIZE && col >= 0 && col < GRID_SIZE) {
        grid[row][col] = 255;
        
        const neighbors: [number, number][] = [
          [-1, -1], [-1, 0], [-1, 1],
          [0, -1],           [0, 1],
          [1, -1],  [1, 0],  [1, 1]
        ];
        
        neighbors.forEach(([dr, dc]: [number, number]) => {
          const newRow = row + dr;
          const newCol = col + dc;
          if (newRow >= 0 && newRow < GRID_SIZE && newCol >= 0 && newCol < GRID_SIZE) {
            grid[newRow][newCol] = Math.max(grid[newRow][newCol], 77);
          }
        });
        
        drawGrid();
      }
    }
    
    function stopDrawing(): void {
      isDrawing = false;
    }
    
    function clearGrid(): void {
      grid = Array(GRID_SIZE).fill(null).map(() => Array(GRID_SIZE).fill(0));
      drawGrid();
    }
    
    async function getPrediction(): Promise<void> {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({pixels: grid})
       })
       neural_network_response.prediction = (await response.json()).message;

    }
    
    function handleTouchStart(event: TouchEvent): void {
      event.preventDefault();
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent('mousedown', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      startDrawing(mouseEvent);
    }
    
    function handleTouchMove(event: TouchEvent): void {
      event.preventDefault();
      const touch = event.touches[0];
      const mouseEvent = new MouseEvent('mousemove', {
        clientX: touch.clientX,
        clientY: touch.clientY
      });
      draw(mouseEvent);
    }
    
    function handleTouchEnd(event: TouchEvent): void {
      event.preventDefault();
      stopDrawing();
    }
  </script>
  
  <div class="mnist-container">
    <div class="canvas-container">
      <canvas
        bind:this={canvas}
        width={CANVAS_SIZE}
        height={CANVAS_SIZE}
        onmousedown={startDrawing}
        onmousemove={draw}
        onmouseup={() => {isDrawing = false; getPrediction()}}
        onmouseleave={stopDrawing}
        ontouchstart={handleTouchStart}
        ontouchmove={handleTouchMove}
        ontouchend={handleTouchEnd}
      ></canvas>
    </div>
    
    <div class="controls">
      <button onclick={clearGrid}>Clear Grid</button>
    </div>
  </div>
  
  <style>
    :global(body) {
      background-color: #111;
    }

    .mnist-container {
      max-width: min-content;
      padding: 20px;
      font-family: Arial, sans-serif;
    }
    
    .canvas-container {
      display: flex;
      justify-content: center;
      margin: 20px 0;
      border: 1px solid white;
      border-radius: 4px;
      background: #111;
    }
    
    canvas {
      cursor: crosshair;
      display: block;
    }
    
    .controls {
      display: flex;
      gap: 10px;
      justify-content: center;
      margin: 20px 0;
    }
    
    button {
      padding: 10px 20px;
      font-size: 14px;
      border: none;
      border-radius: 4px;
      cursor: pointer;
      transition: background-color 0.2s;
    }
    
    button:first-child {
      background-color: #ff6b6b;
      color: white;
    }
    
    button:first-child:hover {
      background-color: #ff5252;
    }
    
    button:last-child {
      background-color: #4CAF50;
      color: white;
    }
    
    button:last-child:hover {
      background-color: #45a049;
    }
  </style>