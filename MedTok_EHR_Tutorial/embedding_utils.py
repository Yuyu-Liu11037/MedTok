import json
import numpy as np
import torch
from typing import Dict, List, Union

def load_embeddings_from_json(json_path: str) -> np.ndarray:
    """
    从JSON文件加载嵌入向量并转换为numpy数组格式
    
    Args:
        json_path: JSON文件路径，格式为 {"code": [embedding_vector], ...}
    
    Returns:
        numpy数组，形状为 (num_codes, embedding_dim)
    """
    print(f"Loading embeddings from {json_path}...")
    
    with open(json_path, 'r') as f:
        code2embeddings = json.load(f)
    
    # 获取所有代码和对应的嵌入
    codes = list(code2embeddings.keys())
    embeddings = list(code2embeddings.values())
    
    # 转换为numpy数组
    embeddings_array = np.array(embeddings, dtype=np.float32)
    
    print(f"Loaded {len(codes)} codes with embedding dimension {embeddings_array.shape[1]}")
    print(f"Sample codes: {codes[:5]}")
    
    return embeddings_array

def create_code_to_index_mapping(json_path: str) -> Dict[str, int]:
    """
    创建代码到索引的映射
    
    Args:
        json_path: JSON文件路径
    
    Returns:
        字典，键为代码，值为索引
    """
    with open(json_path, 'r') as f:
        code2embeddings = json.load(f)
    
    codes = list(code2embeddings.keys())
    return {code: idx for idx, code in enumerate(codes)}

def load_embeddings_with_mapping(json_path: str) -> tuple:
    """
    加载嵌入向量和代码映射
    
    Returns:
        (embeddings_array, code_to_index_mapping)
    """
    embeddings = load_embeddings_from_json(json_path)
    code_mapping = create_code_to_index_mapping(json_path)
    return embeddings, code_mapping
