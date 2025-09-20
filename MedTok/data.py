import json
import numpy as np

# 读取 JSON
with open("code2embeddings.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# data 可能是 {code: [embedding vector]} 的字典
# 比如 {"A00": [0.1, 0.2, ...], "A01": [0.3, 0.4, ...]}
# 你需要把它转成 numpy-friendly 的结构
# 保存整个 dict 也行（dtype=object），但通常 embeddings 会转成 matrix

# 方法1：直接保存 dict
np.save("embedding.npy", data, allow_pickle=True)