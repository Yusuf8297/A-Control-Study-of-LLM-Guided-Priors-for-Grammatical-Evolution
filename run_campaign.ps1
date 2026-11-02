# full campaign (resumable). -Processes 0 means all cores but one; -Matched runs the matched design.
param([int]$Processes = 0, [switch]$Smoke, [switch]$Matched)
$py = ".\.venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
$flags = @()
if ($Matched) { $flags += "--matched" }
if ($Smoke) {
    & $py run_fewshot_parallel.py 2 --smoke @flags
} elseif ($Processes -gt 0) {
    & $py run_fewshot_parallel.py $Processes @flags
} else {
    & $py run_fewshot_parallel.py @flags
}
