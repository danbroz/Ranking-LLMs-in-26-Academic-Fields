#!/bin/bash

# Stop any running Ollama instances
sudo systemctl stop ollama
pkill ollama

ports=(11434 11435 11436 11437 11438)

# Start Ollama instances with isolated model folders and custom ports
for port in "${ports[@]}"; do
    OLLAMA_HOST=0.0.0.0:$port OLLAMA_MODELS="/home/dan/.ollama-$port" nohup ollama serve > ollama_$port.log 2>&1 &
    sleep 5  # Give each instance time to initialize
done

# Pull gemma3:4b model into each Ollama instance
for port in "${ports[@]}"; do
    OLLAMA_HOST=localhost:$port ollama pull gemma3:4b
done

