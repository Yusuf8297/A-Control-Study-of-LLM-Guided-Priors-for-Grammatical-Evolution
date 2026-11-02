# one-off setup: virtual environment + dependencies
$ErrorActionPreference = "Stop"
py -3 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\pip.exe install -r requirements.txt
Write-Host ""
Write-Host "done. quick check:  .\.venv\Scripts\python.exe run_fewshot_parallel.py 2 --smoke"
