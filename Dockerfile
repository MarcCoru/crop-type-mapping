FROM pytorch/pytorch:latest

#RUN apt-get update && apt-get install -y \
#    python3-pip

# dependency of tslearn
RUN pip install --no-cache-dir Cython
RUN pip install tslearn numpy scikit-learn

COPY . /workspace/

CMD [ "python", "./train.py" ]