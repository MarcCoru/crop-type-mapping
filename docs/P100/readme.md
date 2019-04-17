# P100

local
```
conda env export > /tmp/environment.yml
rsync -avz -e "ssh" /tmp/environment.yml ubuntu@10.155.47.236:environment.yml

rsync -avz -P -e "ssh" $HOME/data/BavarianCrops/BavarianCrops.zip ubuntu@10.155.47.236:data/

rsync -avz -e "ssh" /home/marc/projects/crop-type-mapping ubuntu@10.155.47.236: --delete
```


remote
```angular2

    5  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
    6  bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda
    7  conda create -n pytorch
   13  export PATH="/home/ubuntu/miniconda/bin:$PATH"
   14  conda activate
   15  echo ". /home/ubuntu/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
   16  source .bashrc
   17  conda activate pytorch
   24  conda env create -f environment.yml
   26  conda activate pytorch
   27  pip install pyarrow
   28  pip install mkl-random
   29  pip install cython

```