# Tsinghua-Tencent 100K Object Detector Docker Container

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)

This repository is an object detector docker container for Tsinghua-Tencent 100K benchmark.  
It is tested with Nvidia Geforce GTX 1080 Ti. This docker container will try to download tt100k dataset and a weight file from AWS S3 after starting this docker service.  
To access AWS S3, you need to change AWS config file as follows. You can see the AWS config file in src/.aws/config.  

```bash
[default]
aws_access_key_id=GFHFG4GHNFGGHBMKLLDS
aws_secret_access_key=HDaDnIfgkQ8IRvB02gdhslsdfZ+0dfg28pmfhDAc
region=ap-northeast-2
```

## Directories
- You can see the directory structure as follows after this docker service is run.  

/workspace/tt100k/data
- lmdb/test_mean.binaryproto
- test/*.jpg
- annotations.json

/workspace/tt100k/weights
- model.caffemodel
- model.prototxt

/workspace/webapp
- anno_func.py
- app.py
- exifutil.py

/workspace/webapp/.aws
- config

## Docker Build
- Docker repository and tag name is "aibakevision/object-detector-tt100k:gh-latest".  

```bash
./build.sh
```

### Docker Push
- Push docker image

```bash
./push.sh
```

### Docker Run
- Run docker image

```bash
./run.sh
```

## License

This project is made available under the [MIT License](https://github.com/asyncbridge/object-detector-tt100k/blob/master/LICENSE).
