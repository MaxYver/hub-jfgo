# coding:utf8

import torch
import torch.nn as nn
import numpy as np
import random
import json

"""

基于pytorch的网络编写
实现一个网络完成一个简单nlp任务
判断文本中每个字符是否为目标字符集合中的字符，并标注对应位置

"""

# 目标字符集合：需要标注的字符所在位置
TARGET_CHARS = set(['你', '我', '他'])

class TorchModel(nn.Module):
    def __init__(self, vector_dim, sentence_length, vocab):
        super(TorchModel, self).__init__()
        self.embedding = nn.Embedding(len(vocab), vector_dim, padding_idx=0)  # embedding层
        self.rnn = nn.GRU(input_size=vector_dim, hidden_size=vector_dim, batch_first=True)  # 简单GRU
        self.classify = nn.Linear(vector_dim, 1)  # 每个时序步输出一个二分类 logit
        self.loss = nn.CrossEntropyLoss()  # 使用二分类的对数损失

    # 输入x形状：(batch_size, sentence_length)
    # 若有真实标签y，返回loss，否则返回每个位置的概率(0~1)
    def forward(self, x, y=None):
        x = self.embedding(x)  # (B, L) -> (B, L, D)
        x, _ = self.rnn(x)     # (B, L, D)
        logits = self.classify(x).squeeze(-1)  # (B, L)
        if y is not None:
            # y 应该形状为 (B, L)，dtype=float
            return self.loss(logits, y)
        else:
            # 返回每个位置的概率
            return torch.sigmoid(logits)

# 为每个字生成一个标号
# {"a":1, "b":2, "c":3...}
# abc -> [1,2,3]
def build_vocab():
    chars = "你我他defghijklmnopqrstuvwxyz"  # 字符集
    vocab = {"pad": 0}
    for index, char in enumerate(chars):
        vocab[char] = index + 1  # 每个字对应一个序号
    vocab['unk'] = len(vocab)
    return vocab


# 随机生成一个样本
# 生成一个长度为sentence_length的序列，逐位置标注是否在 TARGET_CHARS 中
def build_sample(vocab, sentence_length):
    # 随机从字表选取 sentence_length 个字，可能重复
    x = [random.choice(list(vocab.keys())) for _ in range(sentence_length)]
    # 逐位置标注：若该位置字符属于 TARGET_CHARS，即为正样本
    y = [1 if ch in TARGET_CHARS else 0 for ch in x]
    x = [vocab.get(word, vocab['unk']) for word in x]  # 将字转换成序号，方便 embedding
    return x, y


# 建立数据集
def build_dataset(sample_length, vocab, sentence_length):
    dataset_x = []
    dataset_y = []
    for i in range(sample_length):
        x, y = build_sample(vocab, sentence_length)
        dataset_x.append(x)
        dataset_y.append(y)
    return torch.LongTensor(dataset_x), torch.FloatTensor(dataset_y)


# 建立模型
def build_model(vocab, char_dim, sentence_length):
    model = TorchModel(char_dim, sentence_length, vocab)
    return model


# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model, vocab, sentence_length):
    model.eval()
    x, y = build_dataset(200, vocab, sentence_length)  # 建立200个用于测试的样本
    total = x.numel() // sentence_length  # 样本数
    total_pos = int(y.sum().item())
    print("本次测试集中共有正样本位置数：%d，负样本位置数：%d" % (total_pos, total * sentence_length - total_pos))
    correct, wrong = 0, 0
    with torch.no_grad():
        y_pred = model(x)  # 模型预测，形状：(batch, L)
        # 将预测概率阈值0.5转换为标签
        pred_labels = (y_pred >= 0.5).float()
        # 逐位置对比
        correct = int((pred_labels == y).sum().item())
        wrong = int(y.numel()) - correct
    print("正确预测的总位置数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 10  # 训练轮数
    batch_size = 20  # 每次训练样本个数
    train_sample = 500  # 每轮训练总共训练的样本总数
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
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
            x, y = build_dataset(batch_size, vocab, sentence_length) # 构造一组训练样本
            optim.zero_grad()  # 梯度归零
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optim.step()  # 更新权重

            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model, vocab, sentence_length)  # 测试本轮模型结果
        log.append([acc, np.mean(watch_loss)])

    # 保存模型
    torch.save(model.state_dict(), "model.pth")
    # 保存词表
    writer = open("vocab.json", "w", encoding="utf8")
    writer.write(json.dumps(vocab, ensure_ascii=False, indent=2))
    writer.close()
    return


# 使用训练好的模型做预测
def predict(model_path, vocab_path, input_strings):
    char_dim = 20  # 每个字的维度
    sentence_length = 6  # 样本文本长度
    vocab = json.load(open(vocab_path, "r", encoding="utf8"))  # 加载字符表
    model = build_model(vocab, char_dim, sentence_length)  # 建立模型
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    model.eval()
    results = []
    for input_string in input_strings:
        # 将输入序列化，长度补齐至 sentence_length
        seq = [vocab.get(ch, vocab['unk']) for ch in input_string[:sentence_length]]
        if len(seq) < sentence_length:
            seq += [vocab['pad']] * (sentence_length - len(seq))
        x = torch.LongTensor([seq])  # (1, L)
        with torch.no_grad():
            prob = model(x)  # (1, L)，概率
        probs = prob.squeeze(0).tolist()  # List[float]
        # 输出每个位置的字符、概率和预测标签
        for i in range(sentence_length):
            ch = input_string[i] if i < len(input_string) else 'PAD'
            p = float(probs[i])
            label = int(p >= 0.5)
            # 收集结果
        results.append((input_string, probs))
    # 打印结果
    for input_string, probs in results:
        print("输入：%s" % input_string)
        for i, p in enumerate(probs):
            ch = input_string[i] if i < len(input_string) else 'PAD'
            label = int(p >= 0.5)
            print("  位置%d: 字符='%s', 概率=%f, 预测标签=%d" % (i, ch, p, label))
        print("")

if __name__ == "__main__":
    main()
    test_strings = ["fnvf我e", "wz你dfg", "rqwdeg", "n我kwww"]
    predict("model.pth", "vocab.json", test_strings)
