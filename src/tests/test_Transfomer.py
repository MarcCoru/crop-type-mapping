import sys
sys.path.append("..")
sys.path.append("../models")

import unittest
from models.transformer.Models import Transformer, Encoder
from models.TransformerEncoder import TransformerEncoder
from models.transformer.Layers import EncoderLayer
import os
from datasets.BavarianCrops_Dataset import BavarianCropsDataset
import torch
class Transformer_Test(unittest.TestCase):


    def test_TransformerEncoder(self):
        nclasses = 6
        batchsize = 64
        seq_len = 26
        in_channels = 13


        x = torch.rand(batchsize, in_channels, seq_len)

        model = TransformerEncoder(in_channels=in_channels, len_max_seq=2*seq_len,
            d_word_vec=512, d_model=512, d_inner=2048,
            n_layers=6, n_head=8, d_k=64, d_v=64,
            dropout=0.2, nclasses=6)

        logits, *_ = model._logits(x)
        self.assertEqual(tuple(logits.shape), (batchsize, nclasses))

        probas, *_ = model.forward(x)
        self.assertEqual(tuple(probas.shape), (batchsize, nclasses))

        pass


    def test_encoder_module(self):
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

        #x = torch.randint(100,(batchsize, 26),dtype=torch.long)
        x = torch.rand(batchsize, d_model, 26)
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

        y, atts = encoder.forward(src_seq=x, src_pos=src_pos, return_attns=True)

        self.assertEqual(tuple(y.shape), (batchsize, seq, d_model))

if __name__ == '__main__':
    unittest.main()
