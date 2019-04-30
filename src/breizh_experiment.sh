#python train.py -x BreizhCrops_rnn -e 30 -b 128 --dropout 0.5 -w 1 -i 0 --store /data/breizh/ --checkpoint_every_n_epochs 5 --test_every_n_epochs 5
python train.py -x BreizhCrops_transformer -e 30 -b 128 --dropout 0.5 -w 1 -i 0 --store /data/breizh/ --checkpoint_every_n_epochs 5 --test_every_n_epochs 5 --overwrite
