$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
& "$ProjectRoot\.venv\Scripts\python.exe" -m mapd smoke --config "$ProjectRoot\configs\local_smoke.yaml"
& "$ProjectRoot\.venv\Scripts\python.exe" -m pytest "$ProjectRoot\tests"

