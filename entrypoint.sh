#!/bin/bash
# Download dataset
aws s3 sync s3://aibakevision-object-detection/caffe/tt100k/data /workspace/tt100k/data

#Download tt100k weight file
aws s3 sync s3://aibakevision-object-detection/caffe/tt100k/weights /workspace/tt100k/weights
python app.py
