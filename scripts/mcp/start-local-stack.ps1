$ErrorActionPreference = "Stop"

$postgresService = $env:MARAG_POSTGRES_SERVICE_NAME
$neo4jHome = $env:NEO4J_DBMS_HOME
$neo4jJava = $env:NEO4J_JAVA_HOME

$results = [ordered]@{}

if ($env:MARAG_POSTGRES_AUTO_START -eq "true" -and $postgresService) {
    $pgScript = Join-Path $PSScriptRoot "start-postgres-service.ps1"
    $results.postgres = powershell.exe -NoProfile -ExecutionPolicy Bypass -File $pgScript -ServiceName $postgresService
}

if ($env:MARAG_NEO4J_AUTO_START -eq "true" -and $neo4jHome) {
    $neoScript = Join-Path $PSScriptRoot "start-neo4j-dbms.ps1"
    $results.neo4j = powershell.exe -NoProfile -ExecutionPolicy Bypass -File $neoScript -Neo4jDbmsHome $neo4jHome -JavaHome $neo4jJava
}

$checkScript = Join-Path $PSScriptRoot "check-local-stack.ps1"
$results.check = powershell.exe -NoProfile -ExecutionPolicy Bypass -File $checkScript

$results | ConvertTo-Json -Depth 10
