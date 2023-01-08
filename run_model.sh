#!/bin/bash

# Convert the jupyter notebook to a .py file
jupyter nbconvert --to script model.ipynb

# Build the Docker image
docker build -t my-model .

# Run the Docker image
docker run my-model

