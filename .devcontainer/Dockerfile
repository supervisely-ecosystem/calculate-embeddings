FROM nvidia/cuda:12.1.0-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt update && apt install python3-pip -y
RUN apt-get install -y git
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

RUN pip3 install torch==1.11.0 torchvision==0.12.0

RUN pip3 install transformers==4.33.2 timm==0.9.5 scikit-learn==1.3.1 umap-learn==0.5.4

# Download metaclip base model
RUN python3 -c 'import transformers; transformers.AutoModel.from_pretrained("facebook/metaclip-b16-fullcc2.5b")'

RUN pip3 install supervisely==6.73.314
RUN pip3 install fastapi==0.109.0 bokeh==3.1.1
LABEL python_sdk_version=6.73.314

RUN pip3 install ruamel.yaml==0.17.21

RUN apt-get update

RUN apt-get -y install curl

RUN apt -y install wireguard iproute2
RUN apt-get -y install wget
RUN apt-get install nano

LABEL "role"="development"
