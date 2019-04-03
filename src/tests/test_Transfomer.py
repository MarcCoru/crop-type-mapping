import sys
sys.path.append("..")
sys.path.append("../models")

import unittest
from models.transformer.Models import Transformer, Encoder
from models.transformer.Layers import EncoderLayer
import os
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
class Transformer_Test(unittest.TestCase):

    def test_init(self):
        n_src_vocab=200
        n_tgt_vocab=100
        len_max_seq=52
        d_word_vec = 512
        d_model = 512
        d_inner = 2048
        n_layers = 6
        n_head = 8
        d_k = 64
        d_v = 64
        dropout = 0.1

        batchsize = 64

        #encoder = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

        x = torch.randint(100,(batchsize, 26),dtype=torch.long)
        seq = x.shape[1]
        src_pos = torch.arange(1,seq+1, dtype=torch.long).expand(batchsize,seq)
        src_pos[:, seq - 1] = 0

        non_pad_mask = torch.ones(batchsize, len_max_seq, 1,dtype=torch.float32)
        slf_attn_mask = torch.ones(batchsize, len_max_seq, len_max_seq,dtype=torch.uint8)

        #y = encoder(x,non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)


        encoder = Encoder(
            n_src_vocab=n_src_vocab, len_max_seq=len_max_seq,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=d_inner,
            n_layers=n_layers, n_head=n_head, d_k=d_k, d_v=d_v,
            dropout=dropout)

        y = encoder.forward(src_seq=x, src_pos=src_pos)


if __name__ == '__main__':
    unittest.main()
