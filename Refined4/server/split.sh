# Input video
$input = "Whole_Videos/42/42.mp4"

# Get total duration in seconds
$duration = [math]::Floor((ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 $input))

# Loop through time in 3-second steps
for ($t = 0; $t -lt $duration; $t += 3) {
    $index = "{0:D3}" -f ($t / 3)
    $outfile = "Splitted_Videos/42/chunk_$index.mp4"

    ffmpeg -ss $t -i $input -t 3 -c:v libx264 -preset fast -c:a aac -y $outfile
}

# powershell -ExecutionPolicy Bypass -File split.ps1