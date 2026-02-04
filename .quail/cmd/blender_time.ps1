param (
	[string]$blenderPath = "",
	[string]$blenderFile = "",
	[string]$outputPath = "",
	[string]$runnerpy = "",
    [string]$texturePath = "C:/Projects/ATS-GPU/cycles/textures/1_HR.png"
)


# Call a python script to rearrange the scene. Discard Input.
$null = & cmd.exe /c "${blenderPath}" --background "${blenderFile}.blend" --python "${runnerpy}" -- $texturePath

# Call the rendering process and render.
$output = & cmd.exe /c "${blenderPath}" --background "${blenderFile}.blend" -o "${outputPath}\render_####.png" -f 1

# Only get the final render-time line.
$last = $output | Select-Object -Last 4 | Select-Object -First 1

Write-Host $last

# Get Minutes, Seconds and print
#if ($last -match 'Time:\s+(\d{2}):(\d{2})\.(\d{2})') {
#	$hours 		= [int]$matches[1]
#    $minutes 	= [int]$matches[2]
#    $seconds 	= [int]$matches[3]
#
#    Write-Host "HMS: $hours : $minutes : $seconds"
#}



# Print it.
#Write-Output "minutes:seconds" $last

# To Pass arguments to the runner script.
# cmd.exe /c "${blenderPath}" --background "${blenderFile}.blend" -o "${outputPath}\render_####.png" -f 1 --python "${runnerpy}" -- "0.2,0.5,0.8" "Cube" "2,2,1"

#$executionTime = Measure-Command { Your-Command-Here }
#Write-Host "Last command executed in: $($executionTime.TotalSeconds) seconds"

#Push-Location $project_dir
#& ./$project_name.exe
#
#if ( -not $? ) {
#    $msg = $Error[0].Exception.Message
#    Write-Host "Encountered error during Deleting the Folder. Error Message is $msg. Please check."
#}
#
#Pop-Location
