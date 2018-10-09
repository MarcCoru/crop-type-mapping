# pulls a pre-compuled pytorch container (currently 4.1)
FROM pytorch/pytorch:latest

#RUN apt-get update && apt-get install -y \
#    python3-pip

# dependency of tslearn
RUN pip install --no-cache-dir Cython

# other dependencies
RUN pip install tslearn numpy scikit-learn

# copy source code
COPY . /workspace/

# run python train.py when container starts
CMD [ "python", "-u", "./train.py" ]
