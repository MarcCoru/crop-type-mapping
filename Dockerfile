FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime
COPY src /crop-type-mapping/src
COPY requirements.txt /crop-type-mapping/
RUN pip install -r requirements.txt