FROM aibakevision/object-detector-tt100k-base-gpu:cuda8.0-ubuntu16.04-python2.7.12-gh

# Make workspace
# For tsinghua-tencent 100k
RUN mkdir /workspace/tt100k/data
RUN mkdir /workspace/tt100k/weights

# For Webapp 
RUN mkdir /workspace/webapp
ADD entrypoint.sh /workspace/webapp
ADD ./src /workspace/webapp
WORKDIR /workspace/webapp

# Set aws config file path
ARG aws_config_file=/workspace/webapp/.aws/config
ENV AWS_CONFIG_FILE=${aws_config_file}
RUN echo $AWS_CONFIG_FILE

# Install dependencies of python application
RUN pip install --no-cache-dir protobuf

ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

