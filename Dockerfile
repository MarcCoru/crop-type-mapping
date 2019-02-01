FROM pytorch/pytorch:latest
RUN pip install --no-cache-dir Cython
RUN pip install --upgrade pip
RUN pip install numpy scikit-learn pandas visdom ray matplotlib seaborn
RUN pip install git+https://github.com/marccoru/tslearn.git
COPY . /workspace/