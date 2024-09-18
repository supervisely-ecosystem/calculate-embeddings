FROM supervisely/base-py-sdk:6.73.90


ENV DEBIAN_FRONTEND=noninteractive

RUN pip3 install torch==1.11.0+cu113 torchvision==0.12.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113


RUN pip3 install transformers==4.33.2 timm==0.9.5 scikit-learn==1.3.1 umap-learn==0.5.4

# Download metaclip base model
RUN python -c 'import transformers; transformers.AutoModel.from_pretrained("facebook/metaclip-b16-fullcc2.5b")'

LABEL python_sdk_version=6.73.90
