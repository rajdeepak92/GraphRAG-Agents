$ErrorActionPreference = "Stop"

$results = [ordered]@{
    service_stop = "skipped"
    detail = "MARAG_ALLOW_SERVICE_STOP is not true; no services stopped"
}

if ($env:MARAG_ALLOW_SERVICE_STOP -eq "true") {
    $results.detail = "PostgreSQL stop requires MARAG_POSTGRES_ALLOW_STOP=true; Neo4j stop is intentionally not global."

    if ($env:MARAG_POSTGRES_ALLOW_STOP -eq "true" -and $env:MARAG_POSTGRES_SERVICE_NAME) {
        Stop-Service -Name $env:MARAG_POSTGRES_SERVICE_NAME
        $results.postgres = "stop requested for $env:MARAG_POSTGRES_SERVICE_NAME"
    }

    $results.neo4j = "skipped; no MCP-managed Neo4j PID file is available"
}

$results | ConvertTo-Json -Depth 5
