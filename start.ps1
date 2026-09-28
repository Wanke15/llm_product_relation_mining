param([int]$Limit = 200, [switch]$ReviewOnly)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path -LiteralPath '.env')) {
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
}
if (-not $ReviewOnly) {
    uv run python -m relations run --limit $Limit
    if ($LASTEXITCODE -ne 0) {
        Write-Warning 'Run has errors. Open the review page to inspect existing results; use --resume to retry.'
    }
}
uv run python -m relations serve
