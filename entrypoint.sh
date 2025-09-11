#!/bin/sh

# Start ollama
ollama serve & 
sleep 5

# Pull models
ollama pull llama3.2:3b &
sleep 5
ollama pull mxbai-embed-large:latest

# Create ready signal file
touch models_ready
wait