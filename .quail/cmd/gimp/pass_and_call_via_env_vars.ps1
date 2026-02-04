param (
	[string]$gimp_path = "",
	[string]$python_script_path = ""
)

for ($i = 0; $i -lt $args.Count; $i++) {
    $varName = "GIMP_ATS_TEMP_$i"
    $varValue = $args[$i]
    
    Set-Item -Path "env:$varName" -Value $varValue

    Write-Host "Set environment variable $varName = $varValue"
}

$command = "`"$gimp_path`" --quit --batch-interpreter python-fu-eval -i -b - < `"$python_script_path`""
cmd /c $command
