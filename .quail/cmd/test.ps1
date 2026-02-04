param (
    [string]$ats_profile = "eexe_cpu",
    [string]$hrTex = "textures/0_HR.png",
    [string]$lrTex = "res/textures/nearest/0_320x320x3.png",
    [string]$upTex = "res/textures/upscale/NEDI0_320x320x3.png",
    [string]$folder ="C:/Projects/Adaptive-Texture-Sampling",
	[int]$iterations = 1
)

$inputCycles = "$folder/out/render_0001.png"
$diffImage = "$folder/out/output_diff.png"
$inputATS = "$folder/out/output.png"

function Get-Median {
    param([double[]]$numbers)

    if ($numbers.Count -eq 0) {
        return $null
    }

    $sorted = $numbers | Sort-Object
    $count = $sorted.Count
    $middle = [math]::Floor(($count - 1) / 2)

    if ($count % 2) {
        # Odd count - return middle element
        return $sorted[$middle]
    } else {
        # Even count - average middle two elements
        return (($sorted[$middle] + $sorted[$middle + 1]) / 2)
    }
}

$spinnerInterval = 100  # milliseconds
$spinner = @('-', '\', '|', '/')
#$programPath = "quail -o ats blender 1"

$list_seconds = New-Object System.Collections.ArrayList
$sum_seconds = 0

[Console]::CursorVisible = $false

for ($i = 1; $i -le $iterations; $i++) {
    # Start the executable as a background job
    $job = Start-Job -ScriptBlock {
        param ($hr)
        & quail -o ats blender 1 $ $hr
    } -ArgumentList $hrTex

    # Show spinner until the job finishes
    while ($job.State -eq 'Running') {
        foreach ($frame in $spinner) {
            if ($job.State -ne 'Running') { break }
            Write-Host -NoNewline "`r$frame Running iteration $i of $iterations  "
            Start-Sleep -Milliseconds $spinnerInterval
        }
    }

    # Wait for job to finish and receive output/errors (optional)
    $output = Receive-Job $job
    Remove-Job $job

    #Write-Host "`r$output"

    # Get Hours, Minutes, Seconds
    if ($output -match 'Time: (\d{2}):(\d+\.\d+).*Saving: (\d{2}):(\d+\.\d+)') {
        $minutes = [int]$matches[1]
        $seconds = [double]$matches[2]

        $seconds += ($minutes * 60)
        $list_seconds.Add($seconds) | Out-Null
        $sum_seconds += $seconds
    } else {
        Write-Host "match ERROR!"
    }
    #

}

# Get all stats.
$average = $sum_seconds / $iterations
$minmax  = $list_seconds | Measure-Object -Minimum -Maximum
$median  = Get-Median -numbers $list_seconds
$mid     = $minmax.Minimum + (($minmax.Maximum - $minmax.Minimum) / 2)

[Console]::CursorVisible = $true
Write-Host "`rDone! Cycles -> min:$($minmax.Minimum) max:$($minmax.Maximum) mid:$mid average:$average, median:$median"

#
# Blender done. Do ATS.
#

[Console]::CursorVisible = $false

$list_seconds.clear()
$sum_seconds = 0

for ($i = 1; $i -le $iterations; $i++) {
    # Start the executable as a background job
    $job = Start-Job -ScriptBlock {
        param ($ats_profile, $lrTex, $upTex)
        & quail -o ats $ats_profile $ "$lrTex $upTex"
    } -ArgumentList $ats_profile, $lrTex, $upTex

    #$lrTex = "C:/Projects/ATS-GPU/project/CPU/res/textures/nearest/0_320x320x3.png"
    #$upTex = "C:/Projects/ATS-GPU/project/CPU/res/textures/upscale/NEDI0_320x320x3.png"
    #quail -o ats ccpu $ ",C:/Projects/ATS-GPU/project/CPU/res/textures/nearest/0_320x320x3.png C:/Projects/ATS-GPU/project/CPU/res/textures/upscale/NEDI0_320x320x3.png"

    # Show spinner until the job finishes
    while ($job.State -eq 'Running') {
        foreach ($frame in $spinner) {
            if ($job.State -ne 'Running') { break }
            Write-Host -NoNewline "`r$frame Running iteration $i of $iterations  "
            Start-Sleep -Milliseconds $spinnerInterval
        }
    }

    # Wait for job to finish and receive output/errors (optional)
    $output = Receive-Job $job
    Remove-Job $job

    # Get Hours, Minutes, Seconds
        $match = [regex]::Match($output, '\[(\d+\.\d+)\]')
        $seconds = [float]$match.Groups[1].Value

        $list_seconds.Add($seconds) | Out-Null
        $sum_seconds += $Seconds
    #

}

# Get all stats.
$average = $sum_seconds / $iterations
$minmax  = $list_seconds | Measure-Object -Minimum -Maximum
$median  = Get-Median -numbers $list_seconds
$mid     = $minmax.Minimum + (($minmax.Maximum - $minmax.Minimum) / 2)

[Console]::CursorVisible = $true
Write-Host "`rDone! ATS    -> min:$($minmax.Minimum) max:$($minmax.Maximum) mid:$mid average:$average, median:$median"

#
# 3. Generate Difference Image.
# 

Write-Host "Generating difference image..."

# convert rgba to rgb
magick $inputCycles -background white -alpha remove -alpha off $inputCycles
magick $inputATS -background white -alpha remove -alpha off $inputATS

$null = quail -o ats gimp diff $ "$inputCycles $inputATS $diffImage"
Write-Host "Difference image done."


# 4. Generate Difference Numbers. [median, average, low, high]
# 

Write-Host "`nGenerating comparison..."
quail -o ats ccmp # | Out-Null
Write-Host "`rComparison done."
