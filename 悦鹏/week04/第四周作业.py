Dict = {"经常":0.1,
        "经":0.05,
        "有":0.1,
        "常":0.001,
        "有意见":0.1,
        "歧":0.001,
        "意见":0.2,
        "分歧":0.2,
        "见":0.05,
        "意":0.05,
        "见分歧":0.05,
        "分":0.1}


def all_possible_cuts(sentence, word_dict):
    n = len(sentence)
    result = []
    def dfs(start, path):
        # 如果已处理完整个句子，将路径加入结果
        if start == n:
            result.append(path.copy())
            return
        # 尝试所有可能的结束位置（从start+1到n）
        for end in range(start + 1, n + 1):
            current_word = sentence[start:end]
            if current_word in word_dict:
                path.append(current_word)
                dfs(end, path)
                path.pop()
    dfs(0, [])
    return result

sentence = "经常有意见分歧"
target = all_possible_cuts(sentence, Dict)
print(target)

