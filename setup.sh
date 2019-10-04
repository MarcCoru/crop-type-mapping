wget https://repo.anaconda.com/archive/Anaconda3-2019.07-Linux-x86_64.sh
bash Anaconda3-2019.07-Linux-x86_64.sh -b
conda
conda create -n pytorch pip
conda activate pytorch
pip install --no-cache-dir Cython
pip install numpy psutil
pip install --upgrade pip
pip install git+https://github.com/marccoru/tslearn.git
pip install -r requirements.txt
