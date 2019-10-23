#!/bin/bash
docker run --runtime=nvidia -p 5003:5003 -d aibakevision/object-detector-tt100k:gh-latest
