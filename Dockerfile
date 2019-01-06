# pulls a pre-compuled pytorch container (currently 4.1)
FROM pytorch/pytorch:latest

#RUN apt-get update && apt-get install -y \
#    python3-pip

# dependency of tslearn
RUN pip install --no-cache-dir Cython

RUN pip install --upgrade pip
# other dependencies
RUN pip install tslearn numpy scikit-learn pandas visdom ray matplotlib seaborn

# copy source code
COPY . /workspace/