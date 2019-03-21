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
        len_max_seq=100
        d_word_vec = 512
        d_model = 512
        d_inner = 2048
        n_layers = 6
        n_head = 8
        d_k = 64
        d_v = 64
        dropout = 0.1

        batchsize = 1

        encoder = EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)

        x = torch.rand(batchsize, len_max_seq, d_word_vec)
        non_pad_mask = torch.ones(batchsize, len_max_seq, d_word_vec,dtype=torch.uint8)
        slf_attn_mask = torch.ones(batchsize, len_max_seq, d_word_vec,dtype=torch.uint8)

        y = encoder(x,non_pad_mask=non_pad_mask, slf_attn_mask=slf_attn_mask)


if __name__ == '__main__':
    unittest.main()
