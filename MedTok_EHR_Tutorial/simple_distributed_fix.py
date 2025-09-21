#!/usr/bin/env python3
"""
简单的分布式训练修复方案
"""

import os
import pickle
import torch
import fcntl
import time

def load_dataset_with_distributed_support(args, params):
    """
    支持分布式训练的数据加载函数
    只在rank 0进程加载数据，其他进程从缓存加载
    """
    
    dataset_name = params['dataset']
    task = params['task']
    max_visits = args.max_visits
    
    # 创建缓存文件路径
    cache_path = f"/tmp/dataset_cache_{dataset_name}_{task}_{args.debug_patients if args.debug else 'full'}.pkl"
    
    # 检查是否在分布式环境中
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        
        print(f"分布式训练: Rank {rank}/{world_size}")
        
        if rank == 0:
            # Rank 0: 加载数据并保存到缓存
            print("Rank 0: 开始加载数据集...")
            dataset = _load_dataset_direct(args, params)
            
            # 保存到缓存
            with open(cache_path, 'wb') as f:
                pickle.dump(dataset, f)
            print(f"Rank 0: 数据集已保存到缓存: {cache_path}")
            
        else:
            # 其他进程: 等待并加载缓存
            print(f"Rank {rank}: 等待数据集加载...")
            torch.distributed.barrier()  # 等待rank 0完成
            
            with open(cache_path, 'rb') as f:
                dataset = pickle.load(f)
            print(f"Rank {rank}: 从缓存加载数据集完成")
        
        # 同步所有进程
        torch.distributed.barrier()
        
    else:
        # 检查是否有多GPU但未初始化分布式环境
        gpu_count = torch.cuda.device_count()
        if gpu_count > 1:
            print(f"检测到 {gpu_count} 张GPU，但未初始化分布式环境")
            print("PyTorch Lightning 将自动处理多GPU训练")
            
            # 使用文件锁确保只有一个进程加载数据
            lock_path = cache_path + ".lock"
            dataset = None
            
            # 尝试获取锁并加载数据
            max_retries = 30  # 最多等待30秒
            retry_count = 0
            
            while dataset is None and retry_count < max_retries:
                try:
                    # 检查是否已经有缓存文件
                    if os.path.exists(cache_path):
                        print("发现缓存文件，直接加载...")
                        with open(cache_path, 'rb') as f:
                            dataset = pickle.load(f)
                        print(f"从缓存加载数据集完成，样本数量: {len(dataset)}")
                        break
                    
                    # 尝试获取锁
                    with open(lock_path, 'w') as lock_file:
                        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        
                        # 再次检查缓存文件（可能在等待锁的过程中被其他进程创建）
                        if os.path.exists(cache_path):
                            print("等待期间发现缓存文件，直接加载...")
                            with open(cache_path, 'rb') as f:
                                dataset = pickle.load(f)
                            print(f"从缓存加载数据集完成，样本数量: {len(dataset)}")
                        else:
                            print("获取锁成功，开始加载数据集...")
                            dataset = _load_dataset_direct(args, params)
                            # 保存到缓存供后续进程使用
                            with open(cache_path, 'wb') as f:
                                pickle.dump(dataset, f)
                            print(f"数据集已保存到缓存: {cache_path}")
                        break
                        
                except (IOError, OSError):
                    # 无法获取锁，等待并重试
                    retry_count += 1
                    print(f"等待其他进程完成数据加载... ({retry_count}/{max_retries})")
                    time.sleep(1)
            
            # 清理锁文件
            if os.path.exists(lock_path):
                os.remove(lock_path)
                
            if dataset is None:
                raise RuntimeError("无法加载数据集，可能所有进程都在等待")
                
        else:
            print("单GPU模式: 开始加载数据集...")
            dataset = _load_dataset_direct(args, params)
    
    return dataset

def _load_dataset_direct(args, params):
    """直接加载数据集的函数"""
    from load_data import PatientEHR
    
    dataset_name = params['dataset']
    task = params['task']
    max_visits = args.max_visits
    
    print("**********Start to load patient EHR data**********")
    debug_limit = args.debug_patients if args.debug else None
    
    patient = PatientEHR(
        dataset=dataset_name, 
        split='random', 
        visit_num_th=2, 
        max_visit_th=max_visits, 
        task=task, 
        remove_outliers=True, 
        debug_limit=debug_limit
    )
    
    dataset = patient.patient_ehr_data
    print(f"数据集加载完成，样本数量: {len(dataset)}")
    
    return dataset

def cleanup_dataset_cache(dataset_name, task, debug_patients=None):
    """清理数据集缓存"""
    cache_path = f"/tmp/dataset_cache_{dataset_name}_{task}_{debug_patients if debug_patients else 'full'}.pkl"
    if os.path.exists(cache_path):
        os.remove(cache_path)
        print(f"已清理缓存文件: {cache_path}")

# 使用示例
if __name__ == "__main__":
    # 测试代码
    import argparse
    
    # 模拟参数
    args = argparse.Namespace()
    args.debug = True
    args.debug_patients = 100
    args.max_visits = 100
    
    params = {
        'dataset': 'MIMIC_III',
        'task': 'lenofstay'
    }
    
    # 测试数据加载
    dataset = load_dataset_with_distributed_support(args, params)
    print(f"测试完成，数据集大小: {len(dataset)}")
    
    # 清理缓存
    cleanup_dataset_cache(params['dataset'], params['task'], args.debug_patients)
