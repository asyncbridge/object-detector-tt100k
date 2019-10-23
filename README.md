# Tsinghua-Tencent 100K Object Detector Docker Container

[![Software License](https://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat-square)](LICENSE)

This repository is an object detector docker container for Tsinghua-Tencent 100K benchmark. This docker container uses Caffe deep learning framework for TT100K  and Flask for REST API. It is tested with Nvidia Geforce GTX 1080 Ti. This docker container will try to download tt100k dataset and a weight file from AWS S3 after starting this docker service. To access AWS S3, you need to change AWS config file as follows. You can see the AWS config file in src/.aws/config. To run this container, you need to install Docker CE 18.06.2-ce and Nvidia Docker2.

```bash
[default]
aws_access_key_id=GFHFG4GHNFGGHBMKLLDS
aws_secret_access_key=HDaDnIfgkQ8IRvB02gdhslsdfZ+0dfg28pmfhDAc
region=ap-northeast-2
```

- AWS S3 Directory Structure  
Your expected AWS S3 directory structure is as follows.  

```
your-site
- caffe
  - tt100k
    - data
      - lmdb
        . test_mean.binaryproto
      - test
        . *.jpg
      . annotations.json
    - weights
      . model.caffemodel
      . model.prototxt
```

- entrypoint.sh  
The test data of tt100k and the weight file will be downloaded after running this docker service.  

```
aws s3 sync s3://your-site/caffe/tt100k/data /workspace/tt100k/data
aws s3 sync s3://your-site/caffe/tt100k/weights /workspace/tt100k/weights
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

### Detect REST API   
- Request  
if embed_image request parameter is true, detected_embed_image field of JSON object will be created.  
The detected_embed_image field has a detected image and the image format is png which is encoded by base64.  

| Request Spec. | Value                                  |
|---------------|----------------------------------------|
| Method        | POST                                   |
| Path          | /detect_object                         |
| Request Parameter  | embed_image<br>(true or false) |
| Request Body  | submitImageFile<br>(multiparts/form-data) |

- Response

| Response Spec.   | Value                                                                              |
|------------------|------------------------------------------------------------------------------------|
| HTTP Status Code | 200: Success<br>400: Bad Request Error<br>412: Precondition Failed<br>(File Upload Error) |
| Response body    | JSON                                                                               |                                                                            |                                                                   |

### Detect API Test Example
- Detect API Request
```
POST http://localhost:5003/detect_object HTTP/1.1
…
content-type: multipart/form-data; boundary=--------------------------836206036305278683940222
content-length: 1007015
Connection: keep-alive
----------------------------836206036305278683940222
Content-Disposition: form-data; name="submitImageFile"; filename="2.jpg"
Content-Type: image/jpeg
…
```
- Detect API Response
```
HTTP/1.1 200 OK
Content-Length: 1024
Content-Type: application/json
Server: TornadoServer/5.1.1
Connection: Keep-Alive
…
{
  "msg": "success",  
  "status_code": 200,
  "result": {
      "data": [
	       {"detected_results": 
		   {"imgs": {"0": {"objects": [{"category": "il90", 
					        "score": 648.0, 
						"bbox": {"xmin": 1056.0, 
						"ymin": 768.0, 
						"ymax": 822.0, 
						"xmax": 1105.5}}
					]}}}}, {"elapsed_time": 6.22083306312561}
     ]
  }
}
```

## License

This project is made available under the [MIT License](https://github.com/asyncbridge/object-detector-tt100k/blob/master/LICENSE).
