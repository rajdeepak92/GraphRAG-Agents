param(
    [Parameter(Mandatory=$true)]
    [string]$Neo4jDbmsHome,

    [string]$JavaHome,

    [ValidateSet("console", "start")]
    [string]$StartMode = "console"
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path $Neo4jDbmsHome)) {
    throw "NEO4J_DBMS_HOME not found: $Neo4jDbmsHome"
}

$neo4jBat = Join-Path $Neo4jDbmsHome "bin\neo4j.bat"
if (-not (Test-Path $neo4jBat)) {
    throw "neo4j.bat not found under DBMS home: $neo4jBat"
}

if ($JavaHome) {
    if (-not (Test-Path $JavaHome)) {
        throw "NEO4J_JAVA_HOME not found: $JavaHome"
    }
    $env:JAVA_HOME = $JavaHome
    $env:Path = "$JavaHome\bin;$env:Path"
}

Start-Process `
    -FilePath $neo4jBat `
    -ArgumentList $StartMode `
    -WorkingDirectory $Neo4jDbmsHome `
    -WindowStyle Hidden

[ordered]@{
    status = "started"
    start_mode = $StartMode
    neo4j_dbms_home = $Neo4jDbmsHome
    java_home = $env:JAVA_HOME
} | ConvertTo-Json
