<script>
    import  { Bar } from 'svelte5-chartjs';
  
    import {
      Chart,
      Title,
      Tooltip,
      Legend,
      BarElement,
      CategoryScale,
      LinearScale,
    } from 'chart.js';
	import { neural_network_response } from './prediction.svelte';

    Chart.register(
      Title,
      Tooltip,
      Legend,
      BarElement,
      CategoryScale,
      LinearScale
    );

    let data = $derived({
    labels: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    datasets: [
        {
        label: '% of Confidence',
        data: neural_network_response.prediction, 
        backgroundColor: Array(10).fill("rgba(63, 161, 58, 0.8)"),
        borderWidth: 2,
        borderColor: Array(10).fill("rgb(92, 230, 85)"),
    },
    ],
});
</script>
 
<div class = "chart-container" style="flex: 1;">
    <Bar {data} options={{ indexAxis: 'y', maintainAspectRatio: false }} />
</div>

