#!/usr/bin/env bash
cwd=$(dirname "$0"); cd "$cwd"
for model in gemini-2.5-flash gemini-2.5-pro gemini-2.5-flash-lite gemini-2.0-flash gemini-2.0-flash-lite; do
    echo $model
    time vttiro transcribe --verbose -k -e gemini -m $model -i test2.mp4 -o test2.$model.vtt 
done

for model in whisper-1 gpt-4o-transcribe gpt-4o-mini-transcribe; do
    echo $model
    time vttiro transcribe --verbose -k -e openai -m $model -i test2.mp4 -o test2.oai-$model.vtt 
done
