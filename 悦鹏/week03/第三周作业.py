# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中是否有某些特定字符出现

"""


class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        #去掉池化层，加入RNN层，不计算平均
        self.rnn = nn.RNN(vector_dim, vector_dim, batch_first=True)  # 新增RNN层
        self.classify = nn.Linear(vector_dim, 6)  # 输出6个类别（0-5）没见过的就是5
        #去掉sigmoid，使用CrossEntropyLoss 改为多分类交叉熵
        self.loss = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.embedding(x)  # (batch_size, sen_len) -> (batch_size, sen_len, vector_dim)
        x, _ = self.rnn(x)  # (batch_size, sen_len, vector_dim) -> (batch_size, sen_len, vector_dim)
        x = x[:, -1, :]  # 取最后一个时间步的输出 (batch_size, vector_dim)
        y_pred = self.classify(x)  # (batch_size, vector_dim) -> (batch_size, 6)

        if y is not None:
            return self.loss(y_pred, y.squeeze().long())  # 使用交叉熵损失
        else:
            return torch.softmax(y_pred, dim=-1)  # 返回softmax概率


def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  #字符集  自己设置的
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
def build_sample(vocab, sentence_length):
    # 随机从字表选取sentence_length个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]

    # 根据"我"字的位置确定类别
    if "我" in x:
        # "我"出现在第几个位置，就属于第几类
        y = x.index("我")
    else:
        # 没有"我"字时属于第6类（INDEX5）
        y = 5

    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append([y])  # y现在是0-5的整数
    return torch.LongTensor(dataset_x), torch.LongTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
def evaluate(model, vocab, sample_length):
    model.eval()
    x, y = build_dataset(200, vocab, sample_length)

    # 统计各类别数量
    print("各类别样本数量：", [torch.sum(y == i).item() for i in range(6)])

    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，返回softmax概率
        predictions = torch.argmax(y_pred, dim=1)  # 获取预测类别
        y_true = y.squeeze()

        correct = torch.sum(predictions == y_true).item()
        total = len(y_true)
        accuracy = correct / total

    print("正确预测个数：%d, 正确率：%f" % (correct, accuracy))
    return accuracy


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 5  # 文本最大长度为5
    learning_rate = 0.005  # 学习率

    # 建立字表
    vocab = build_vocab()
    # 建立模型
    model = build_model(vocab, char_dim, sentence_length)
    # 选择优化器
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []

    # 训练过程
    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch in range(int(train_sample / batch_size)):
            x, y = build_dataset(batch_size, vocab, sentence_length)  # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存的模型名改为model1.pth 别跟NLPDEMO的重复了
    torch.save(model.state_dict(), "model1.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 5  # 文本最大长度为5
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))
    model = build_model(vocab, char_dim, sentence_length)
    model.load_state_dict(torch.load(model_path))
    x = []
    for input_string in input_strings:
        # 确保输入长度为5，不足则填充
        padded_string = input_string.ljust(sentence_length, " ")[:sentence_length]
        x.append([vocab.get(char, vocab["unk"]) for char in padded_string])

    model.eval()
    with torch.no_grad():
        result = model.forward(torch.LongTensor(x))  # 返回softmax概率

    for i, input_string in enumerate(input_strings):
        probs = result[i]
        predicted_class = torch.argmax(probs).item()
        probability = probs[predicted_class].item()
        print("输入：%s, 预测类别：%d, 概率值：%f" % (input_string, predicted_class, probability))


if __name__ == "__main__":
    main()
    test_strings = ["你我他dy", "uvwxyz", "我他defg", "jklmno"]  # 请在这里添加您的测试字符串
    predict("model1.pth", "vocab.json", test_strings)
