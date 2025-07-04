<script lang="ts">
  import Grid from "../components/Grid.svelte";
  import BarGraph from "../components/BarGraph.svelte";
</script>

<div class = "container">
  <Grid />
  <BarGraph />
</div>

<style>
  .container {
    display: flex;
  }
  @media only screen and (max-width: 800px) {
  .container {
    flex-direction: column;
    align-items: center;
  }
}

</style>