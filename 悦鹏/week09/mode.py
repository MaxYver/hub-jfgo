# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel
"""
建立网络模型结构
"""

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        self.bert = BertModel.from_pretrained(config["bert_path"])
        bert_hidden_size = self.bert.config.hidden_size
        class_num = config["class_num"]
        self.classify = nn.Linear(bert_hidden_size, class_num)#因为是双向LSTM，所以*2；这个线性层是区分有多少类实体标签（class_num这个参数）
        self.crf_layer = CRF(class_num, batch_first=True)#这里也是，转移矩阵的维度跟标签（lable）数量是相关的
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)  #loss采用交叉熵损失

    #当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, target=None):
        attention_mask = (x != 0).long()
        outputs = self.bert(x, attention_mask=attention_mask)

        x = outputs.last_hidden_state


        predict = self.classify(x)  # output shape: (batch_size, seq_len, class_num)

        if target is not None:
            if self.use_crf:
                mask = target.gt(-1) 
                return - self.crf_layer(predict, target, mask, reduction="mean")
            else:
                #(number, class_num), (number)
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))#不用crf的做法 因为交叉熵有形状的要求，需要通过view转换交叉熵的维度--分词那节课有讲
        else:
            if self.use_crf:
                return self.crf_layer.decode(predict)   #decode维特比实现，工具封装好的
            else:
                return predict


def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)


if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)
