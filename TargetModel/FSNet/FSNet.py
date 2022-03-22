"""
Date: 2022-03-03
Author: sunhanwu@iie.ac.cn
Desc: FSNet model Implemented by pytorch
      cite: Liu, Chang, et al. "Fs-net: A flow sequence network for encrypted traffic classification."
            IEEE INFOCOM 2019-IEEE Conference On Computer Communications. IEEE, 2019.
"""

import torch
from torch import nn
import torch.nn.functional as F

class FSNet(nn.Module):
    """
    FSNet model
    """
    def __init__(self, param):
        """
        define some layer
        :param vocab_size: size of the vocab
        :param emb_dim: dim of embedding layer
        :param hidden_size: hidden neural number of encoder and decoder
        :param num_layers: number of gru layers
        :param num_direction: number of gru direction
        """
        super(FSNet, self).__init__()
        self.vocab_size = param['vocab_size']
        self.emb_dim = param['emb_dim']
        self.hidden_size = param['hidden_size']
        self.num_layers = param['num_layers']
        self.num_direction = param['num_direction']
        self.num_class = param['num_class']
        self.sequence_len = param['sequence_len']

        # Embedding layer
        # self.embedding = nn.Embedding(self.vocab_size, self.emb_dim)

        # Encoder layer
        self.encode_gru = nn.GRU(
            # input_size=self.emb_dim,
            input_size=1,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional= True if self.num_direction == 2 else 1,
            batch_first=True)

        # Decoder layer
        self.decode_gru = nn.GRU(
            input_size=self.hidden_size * self.num_layers * self.num_direction,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bidirectional= True if self.num_direction == 2 else 1,
            batch_first=True)

        # Reconstruction_layer
        self.reconstruct = nn.Linear(self.num_direction * self.hidden_size, self.vocab_size)

        # dense layer
        self.linear1 = nn.Linear(self.num_layers * self.num_direction * 4 * self.hidden_size, 64)
        self.linear2 = nn.Linear(64, self.num_class)


    def encode(self, inputs):
        """
        Encoder for FSNet
        :param inputs: batch_x
        :return:
        """
        # embedding_inputs = self.embedding(inputs)
        embedding_inputs = inputs.unsqueeze(2)
        embedding_inputs = embedding_inputs.float()
        # embedding_inputs.shape: (batch_size, sequence_len, embedding_dim)
        encode_inputs1, h_n = self.encode_gru(embedding_inputs)
        # encode_inputs1.shape: (batch_size, sequence_len, hidden_size * num_direction)
        # h_n.shape: (num_layers * num_direction, batch_size, hidden_size)
        output = h_n.permute(1, 0, 2)
        # output.shape: (batch_size, num_layers * num_direction, hidden_size)
        output = torch.reshape(output, [output.shape[0], -1])
        # output.shape: (batch_size, num_layers * num_direction * hidden_size)
        return output


    def decode(self, inputs):
        """
        decoder for fsnet
        :param inputs: ze
        :return:
        """
        inputs_reshape = torch.unsqueeze(inputs, 1).repeat(1, self.sequence_len, 1)
        # inputs_reshape.shape=(batch_size, sequence_len, num_layers * num_direction * hidden_size)
        decode_output, h_n = self.decode_gru(inputs_reshape)
        # decode_output.shape = (batch_size, sequence_len, hidden_size * num_direction)
        # h_n.shape: (num_layers * num_direction, batch_size, hidden_size)
        zd = h_n.permute(1, 0, 2)
        zd = torch.reshape(zd, [zd.shape[0], -1])
        # zd.shape: (batch_size, num_layers * num_direction * hidden_size)
        D = decode_output
        return zd, D



    def reconstruction(self, inputs):
        """

        :param inputs:
        :return:
        """
        # D.shape: (batch_size, sequence_len, hidden_size * num_direction)
        output = self.reconstruct(inputs)
        # output.shape= (batch_size, sequence_len, vocab_size)
        return output

    def dense(self, ze, zd):
        """

        :param ze: output of encode layer
        :param zd: secode part of the decode layer ouput
        :return:
        """
        # ze.shape: (batch_size, num_layers * num_direction * hidden_size)
        # zd.shape: (batch_size, num_layers * num_direction * hidden_size)
        z_dot = ze * zd
        # z_dot.shape: (batch_size, num_layers * num_direction * hidden_size)
        z_minus = torch.abs(ze - zd)
        # z_minus.shape: (batch_size, num_layers * num_direction * hidden_size)
        z = torch.cat((ze, zd, z_dot, z_minus), dim=1)
        # z.shape=(batch_size, num_layers * num_direction * 4 *  hidden_size)
        z = F.selu(self.linear1(z))
        # z.shape=(batch_size, 64)
        z = F.selu(self.linear2(z))
        # z.shape=(batch_size, num_class)
        return z

    def forward(self, inputs):
        z_e = self.encode(inputs)
        z_d, D = self.decode(z_e)
        output = self.dense(z_e, z_d)
        return output