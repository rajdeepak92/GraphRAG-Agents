param(
    [Parameter(Mandatory=$true)]
    [string]$ServiceName
)

$ErrorActionPreference = "Stop"

$svc = Get-Service -Name $ServiceName -ErrorAction SilentlyContinue
if (-not $svc) {
    throw "PostgreSQL service not found: $ServiceName"
}

if ($svc.Status -ne "Running") {
    Start-Service -Name $ServiceName
}

Get-Service -Name $ServiceName | Select-Object Name, Status | ConvertTo-Json
