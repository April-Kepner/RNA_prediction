import math, random
from const import START, STOP

import numpy as np
from collections import defaultdict, OrderedDict
from pprint import pprint

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer
#from paddle.fluid.dygraph.learning_rate_scheduler import LearningRateDecay

class Network(Layer):
    def __init__(self,
                 sequence_vocabulary, bracket_vocabulary,
                 dmodel=128,
                 layers=8,
                 dropout=0.15,
                 ):
        super(Network, self).__init__()
        self.sequence_vocabulary = sequence_vocabulary
        self.bracket_vocabulary = bracket_vocabulary
        self.dropout_rate = dropout
        self.model_size = dmodel
        self.layers = layers

    def forward(self, seq, dot):
        data = [[0,1],[0,1],[0,1],[0,1],[1,0]]
        #data = [[0],[0],[0],[0],[1]]
        weight_data = np.array(data)  # numpy格式的词向量数据
        w_param_attrs = fluid.ParamAttr(
            name="emb_weight",
            learning_rate=5e-4,
            initializer=fluid.initializer.NumpyArrayInitializer(weight_data),
            trainable=True)


        emb_seq = paddle.fluid.embedding(seq, size=(self.sequence_vocabulary.size, self.model_size), is_sparse=True)
        emb_dot = paddle.fluid.embedding(dot, size=(self.bracket_vocabulary.size, self.model_size), is_sparse=True)
        emb = paddle.fluid.layers.concat(input=[emb_seq,emb_dot], axis=1)
        emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
        for _ in range(self.layers):
            emb = paddle.fluid.layers.fc(emb, size=self.model_size*4)    
            fwd, cell  = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*4, use_peepholes=True, is_reverse=False)
            back, cell = paddle.fluid.layers.dynamic_lstm(input=emb, size=self.model_size*4, use_peepholes=True, is_reverse=True)
            emb = paddle.fluid.layers.concat(input=[fwd, back], axis=1)
            emb = paddle.fluid.layers.dropout(emb, dropout_prob=self.dropout_rate)
            emb = paddle.fluid.layers.fc(emb, size=self.model_size, act="relu")
            emb = paddle.fluid.layers.layer_norm(emb)

        ff_out = paddle.fluid.layers.fc(emb, size=2, act="relu")
        emb_2 = fluid.embedding(input=dot, size=(self.bracket_vocabulary.size, 2), param_attr=w_param_attrs, dtype='float32')
        soft_out = paddle.fluid.layers.softmax(ff_out+emb_2, axis=1)
        return soft_out[:,0]


