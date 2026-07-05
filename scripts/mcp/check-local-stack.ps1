param(
    [string]$PostgresServiceName = $env:MARAG_POSTGRES_SERVICE_NAME,
    [string]$PostgresHost = "127.0.0.1",
    [int]$PostgresPort = 5432,
    [string]$Neo4jHost = "127.0.0.1",
    [int]$Neo4jHttpPort = 7474,
    [int]$Neo4jBoltPort = 7687
)

$ErrorActionPreference = "Stop"

function Test-Port {
    param([string]$HostName, [int]$Port)
    $client = New-Object Net.Sockets.TcpClient
    try {
        $async = $client.BeginConnect($HostName, $Port, $null, $null)
        $ok = $async.AsyncWaitHandle.WaitOne(2000, $false)
        if ($ok) {
            $client.EndConnect($async)
        }
        return $ok
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

$result = [ordered]@{
    postgres_service = $null
    postgres_port = Test-Port -HostName $PostgresHost -Port $PostgresPort
    neo4j_http = Test-Port -HostName $Neo4jHost -Port $Neo4jHttpPort
    neo4j_bolt = Test-Port -HostName $Neo4jHost -Port $Neo4jBoltPort
}

if ($PostgresServiceName) {
    $svc = Get-Service -Name $PostgresServiceName -ErrorAction SilentlyContinue
    $result.postgres_service = if ($svc) { $svc.Status.ToString() } else { "not_found" }
}

$result | ConvertTo-Json -Depth 5
