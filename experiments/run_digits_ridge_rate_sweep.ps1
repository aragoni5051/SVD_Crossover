param(
    [int]$Runs = 10,
    [int]$Generations = 300,
    [int]$Population = 20,
    [int]$Elitism = 1,
    [int]$RefineSteps = 2,
    [double]$RefineLr = 0.01,
    [int]$RefineBatchSize = 128,
    [double]$MutationRate = 0.05,
    [double]$LayerDeleteRidgeLambda = 0.001,
    [string]$BaseName = "digits_ridge_layer_delete"
)

$ErrorActionPreference = "Stop"

$rates = @("0.3", "0.5", "0.7", "1.0")
$runName = "${BaseName}_${Runs}x${Generations}_crossoverrate"
$timer = [System.Diagnostics.Stopwatch]::StartNew()

Write-Host "Starting sweep at $(Get-Date)"
Write-Host "Output prefix: results\$runName"

python experiments\compare_evolution_crossover.py `
    --runs $Runs `
    --generations $Generations `
    --population $Population `
    --elitism $Elitism `
    --refine-steps $RefineSteps `
    --refine-lr $RefineLr `
    --refine-batch-size $RefineBatchSize `
    --signed-relu-width-policy mean `
    --crossover-rate 0.3 `
    --layer-delete-ridge-lambda $LayerDeleteRidgeLambda `
    --topology-mutation-rate $MutationRate `
    --node-split-rate $MutationRate `
    --layer-delete-rate $MutationRate `
    --node-delete-rate $MutationRate `
    --methods no_and_same_dim `
    --output-dir "results\$runName`_0.3"

for ($i = 1; $i -lt $rates.Count; $i++) {
    $rate = $rates[$i]
    $outDir = "results\$runName`_$rate"
    $extraArgs = @()

    for ($j = 0; $j -lt $i; $j++) {
        $prevRate = $rates[$j]
        $prevDir = "results\$runName`_$prevRate"
        $extraArgs += @(
            "--extra-curves-csv", "$prevDir\curves.csv",
            "--extra-summary-csv", "$prevDir\summary.csv"
        )
    }

    Write-Host "Starting same-dim crossover rate $rate at $(Get-Date)"

    python experiments\compare_evolution_crossover.py `
        --runs $Runs `
        --generations $Generations `
        --population $Population `
        --elitism $Elitism `
        --refine-steps $RefineSteps `
        --refine-lr $RefineLr `
        --refine-batch-size $RefineBatchSize `
        --signed-relu-width-policy mean `
        --crossover-rate $rate `
        --layer-delete-ridge-lambda $LayerDeleteRidgeLambda `
        --topology-mutation-rate $MutationRate `
        --node-split-rate $MutationRate `
        --layer-delete-rate $MutationRate `
        --node-delete-rate $MutationRate `
        --methods same_dim_only `
        --output-dir $outDir `
        @extraArgs
}

$timer.Stop()
Write-Host "Finished sweep at $(Get-Date)"
Write-Host ("Total elapsed: {0:hh\:mm\:ss}" -f $timer.Elapsed)
Write-Host ("Total seconds: {0:N1}" -f $timer.Elapsed.TotalSeconds)



