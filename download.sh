#!/bin/bash

function downloadBavarianCrops {
  mkdir data
  cd data
  wget https://zenodo.org/record/5707142/files/BavarianCrops.zip
  unzip -o BavarianCrops.zip
  rm BavarianCrops.zip
  cd ..
}

function downloadmodels {
  mkdir models
  cd models
  wget https://zenodo.org/record/5707142/files/models.zip
  unzip -o models.zip
  rm models.zip
  cd ..
}

function downloadnotebookdata {
  mkdir data
  cd data
  wget https://zenodo.org/record/5707142/files/notebookdata.zip
  unzip -o notebookdata.zip
  rm notebookdata.zip
  cd ..
}

function downloaddataduplo {
  mkdir models
  cd models
  wget https://zenodo.org/record/5707142/files/duplo.zip
  unzip -o duplo.zip
  rm duplo.zip
  cd ..
}


if [ "$1" == "dataset" ]; then
    downloadBavarianCrops
elif [ "$1" == "models" ]; then
    downloadmodels
elif [ "$1" == "notebookdata" ]; then
    downloadnotebookdata
elif [ "$1" == "duplo" ]; then
    downloaddataduplo
elif [ "$1" == "all" ]; then
    downloadBavarianCrops
    downloadmodels
    downloadnotebookdata
    downloaddataduplo
else
    echo "please provide 'dataset', 'models','notebookdata', 'duplo', or 'all' as argument"
fi

