import pickle
import json
import random
import shutil
import networkx as nx
from tqdm import tqdm
import numpy as np
import torch
import dgl
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import torch.nn.functional as F
import argparse
import logging
from torch.utils.data import DataLoader as torchDataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.distributed import init_process_group
import pandas as pd
import pickle
import os
from torch.nn.utils.rnn import pad_sequence
from load_data import PatientEHR
from dataloader import PatientDataset, collate
from EHRModel_token import EHRModel
from StandardTransformer import StandardTransformer
import sys
import wandb
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

def ddp_setup(rank, world_size, backend='nccl'):
    #os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1234'
    init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Set deterministic behavior for PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for reproducibility
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def construct_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='MIMIC_IV', choices=['MIMIC_III', 'MIMIC_IV', 'EHRShot'])
    parser.add_argument('--model', type=str, default='Transformer', choices=['MLP', 'Transformer'])
    parser.add_argument('--use_standard_transformer', type=bool, default=False, help='Whether to use standard Transformer instead of EHR-specific Transformer')
    parser.add_argument('--task', type=str, default='readmission', choices=['mortality', 'readmission', 'lenofstay', 'drugrec', 'phenotype', 'new_disease'])
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--num_layers', type=int, default=4, choices=[2,4,6])
    parser.add_argument('--decay_rate', type=float, default=0.01)
    parser.add_argument('--freeze_emb', type=str, default="False")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--alpha', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--beta', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--edge_attn', type=str, default="True", choices=["True", "False"])
    parser.add_argument('--self_attn', type=float, default=0.)
    parser.add_argument('--hyperparameter_search', type=bool, default=False)
    parser.add_argument('--attn_init', type=str, default="False", choices=["True", "False"])
    parser.add_argument('--in_drop_rate', type=float, default=0.)
    parser.add_argument('--out_drop_rate', type=float, default=0.)
    parser.add_argument('--kg_ratio', type=float, default=1.0)
    parser.add_argument('--ehr_feat_ratio', type=float, default=1.0)
    parser.add_argument('--pos_enc_dim', type=int, default=8)
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--mimic_dir_path', type=str)
    parser.add_argument('--save_result_path', type=str, default="task_results")
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--local_rank', type=int)
    parser.add_argument('--memory_bank_size', type=int, default=1024)
    parser.add_argument('--max_visits', type=int, default=100)
    parser.add_argument('--max_medical_code', type=int, default=2000)
    parser.add_argument('--input_dim', type=int, default=64)
    parser.add_argument('--output_dim', type=int, default=64)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--embedding_path', type=str, default='../MedTok/code2embeddings.json')
    parser.add_argument('--use_partial_data', type=int, default=None, help='Number of patients to process for debugging (e.g., 1000). If None, process all data.')
    
    # CPCC Loss parameters
    parser.add_argument('--use_cpcc', type=int, default=0, help='Whether to use CPCC loss (0/1)')
    parser.add_argument('--cpcc_lamb', type=float, default=1.0, help='Lambda weight for CPCC loss')
    parser.add_argument('--cpcc_distance_type', type=str, default='l2', choices=['l2', 'l1', 'cosine', 'poincare'], help='Distance metric for CPCC loss')
    parser.add_argument('--cpcc_center', type=int, default=0, help='Whether to use centering regularization (0/1)')
    parser.add_argument('--cpcc_only', type=int, default=0, help='Whether to use only CPCC loss (no base loss) (0/1)')
    
    # Multiple runs parameters
    parser.add_argument('--num_runs', type=int, default=5, help='Number of runs for averaging results')
    parser.add_argument('--base_seed', type=int, default=42, help='Base seed for random number generation')

    args = parser.parse_args()
    return args

def get_logger(dataset, task, kg, hidden_dim, epochs, lr, decay_rate, dropout, num_layers, exp_dir_path):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s')

    exp_training_log_path = os.path.join(exp_dir_path, "training_logs")
    os.makedirs(exp_training_log_path, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(exp_training_log_path, f'{dataset}_{task}_{hidden_dim}_{epochs}_{lr}_{decay_rate}_{dropout}_{num_layers}.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger

def single_run(args, params, logger, run_id=0):

    dataset_name, task, batch_size, hidden_dim, epochs, lr, weight_decay, dropout, num_layers, decay_rate, alpha, beta, freeze, attn_init, in_drop_rate, kg_ratio, train_ratio, feat_ratio, model_name, pos_enc_dim, debug, mimic_dir_path, save_result_path = \
        params['dataset'], params['task'], params['batch_size'], params['hidden_dim'], params['epochs'], params['lr'], params['weight_decay'], params['dropout'], params['num_layers'], params['decay_rate'], params['alpha'], params['beta'], params['freeze'], params['attn_init'], params['in_drop_rate'], params['kg_ratio'], params['train_ratio'], params['feat_ratio'], params['model'], params['pos_enc_dim'], params['debug'], params['mimic_dir_path'], params['save_result_path']
    
    # Set random seed for reproducibility
    current_seed = args.base_seed + run_id
    set_random_seed(current_seed)
    print(f"Run {run_id + 1}/{args.num_runs}: Random seed set to {current_seed}")
    
    max_visits = args.max_visits
    max_medical_code = args.max_medical_code

    # load dataset
    print("**********Start to load patient EHR data**********")
    patient = PatientEHR(dataset_name, split='random', visit_num_th=2, max_visit_th=max_visits, task=task, remove_outliers=True, use_partial_data=args.use_partial_data)
    dataset = patient.patient_ehr_data
    # [{'patient_id': '10001217', 'birthdate': datetime.datetime(2102, 9, 23, 0, 0), 'deathdate': None, 'gender': 'F', 'ethnicity': 'WHITE', 'conditions_map': [([5826, 6889, 16462, 4048, 16477, 4405, 4406, 3804, 765, 4254],)], 'procedures_map': [([688, 340, 5735],)], 'drugs_map': [([4824, 4978, 1179, 4718, 325, 356, 371, 4612, 368, 329, 321, 1093, 3084, 4608, 3234, 271, 846, 4721, -1, 5317, 3053],)], 'label': 0, 'timestamp_encounter': (datetime.datetime(2157, 11, 18, 22, 56),), 'timestamp_discharge': (datetime.datetime(2157, 11, 25, 18, 0),)}] 

    print("Number of samples: {}".format(len(dataset)))
    filter_out_dataset = []
    for d in dataset:
        if len(d) == 0 or d[0]['label'] == None:
            continue
        else:
            filter_out_dataset.append(d)
    print("Number of samples: {}".format(len(filter_out_dataset)))
    dataset = filter_out_dataset
    print("Number of samples: {}".format(len(dataset)))
    #print(dataset)
    
    if args.task == 'phenotype':
        labels = []
        for i in range(len(dataset)):
            l = dataset[i][0]['label']
            p = [1 if j in l else 0 for j in range(24)]
            labels.append(p)
        labels = np.array(labels)

        column_non_zero_counts = np.sum(labels != 0, axis=0)
        valid_columns = column_non_zero_counts >= 3
        filtered_labels = labels[:, valid_columns]
        labels = filtered_labels
        num_class = labels.shape[-1]
    elif args.task == 'drugrec':
        labels = []
        for i in range(len(dataset)):
            l = dataset[i][0]['label']
            p = [1 if j in l else 0 for j in range(5)]
            labels.append(p)
        labels = np.array(labels)
        column_non_zero_counts = np.sum(labels != 0, axis=0)
        valid_columns = column_non_zero_counts >= 3
        filtered_labels = labels[:, valid_columns]
        labels = filtered_labels
        num_class = labels.shape[-1]
        print(labels.shape)
        print(num_class)
    else:
        labels = [int(dataset[i][0]['label']) for i in range(len(dataset))]
        labels = np.array(labels)
        num_class = np.max(labels) + 1
    
    train_indices, val_indices = train_test_split(
        np.arange(len(dataset)),
        test_size=0.2,  # 20% validation
        stratify=None if args.task in ['phenotype', 'drugrec'] else labels,  # Maintain label distribution
        random_state=42  # For reproducibility
    )
    
    val_indices, test_indices = train_test_split(
        val_indices,
        test_size=0.5,  # 50% of the validation set
        stratify= None if args.task in ['phenotype', 'drugrec'] else labels[val_indices],  # Maintain label distribution
        random_state=42  # For reproducibility
    )
    
    # Create Subsets for train and validation
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    print("Number of samples in train: {}, Number of samples in test: {}, Number of samples in val: {}".format(len(train_dataset), len(test_dataset), len(val_dataset)))

    from torch.utils.data import WeightedRandomSampler

    if args.task in ['phenotype', 'drugrec']:
        sample_weights = np.ones(len(labels))
    else:
        class_counts = torch.bincount(torch.tensor(labels))
        class_weights = 1.0 / class_counts
        sample_weights = class_weights[labels]
    sampler_train = WeightedRandomSampler(sample_weights[train_indices], num_samples=len(labels[train_indices]), replacement=True)
    sampler_val = WeightedRandomSampler(sample_weights[val_indices], num_samples=len(labels[val_indices]), replacement=True)
    sampler_test = WeightedRandomSampler(sample_weights[test_indices], num_samples=len(labels[test_indices]), replacement=True)


    print("**********Data Loader**********")
    train_dataset = PatientDataset(dataset=train_dataset, max_visits=args.max_visits, max_medical_code=args.max_medical_code, task=args.task, labels=labels, embedding_path=args.embedding_path)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, sampler = sampler_train, drop_last=True) 

    val_dataset = PatientDataset(dataset=val_dataset, max_visits=args.max_visits, max_medical_code=args.max_medical_code, task=args.task, labels=labels, embedding_path=args.embedding_path)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, sampler=sampler_val, drop_last=True)

    test_dataset = PatientDataset(dataset=test_dataset, max_visits=args.max_visits, max_medical_code=args.max_medical_code, task=args.task, labels=labels, embedding_path=args.embedding_path)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, collate_fn=collate, num_workers=args.num_workers, sampler=sampler_test, drop_last=True)


    # get model
    print("Getting model...")
    
    if args.use_standard_transformer:
        print("Using Standard Transformer model...")
        model = StandardTransformer(model_name='Transformer', input_dim=args.input_dim, num_heads=args.num_heads, num_layers=args.num_layers, dropout_prob=args.dropout, 
                                   hidden_dim=args.hidden_dim, output_dim=args.output_dim, memory_bank_size=args.memory_bank_size, code_size=21000, lr=args.lr, task=args.task, num_class=num_class,
                                   pre_trained_embedding=args.embedding_path,
                                   use_cpcc=bool(args.use_cpcc), cpcc_lamb=args.cpcc_lamb, cpcc_distance_type=args.cpcc_distance_type, cpcc_center=bool(args.cpcc_center), cpcc_only=bool(args.cpcc_only))
    else:
        print("Using EHR-specific Transformer model...")
        model = EHRModel(model_name='Transformer', input_dim=args.input_dim, num_heads=args.num_heads, num_layers=args.num_layers, dropout_prob=args.dropout, 
                         hidden_dim=args.hidden_dim, output_dim=args.output_dim, memory_bank_size=args.memory_bank_size, code_size=21000, lr=args.lr, task=args.task, num_class=num_class,
                         pre_trained_embedding=args.embedding_path,
                         use_cpcc=bool(args.use_cpcc), cpcc_lamb=args.cpcc_lamb, cpcc_distance_type=args.cpcc_distance_type, cpcc_center=bool(args.cpcc_center))
    
    total_params = sum(param.numel() for param in model.parameters())
    print(total_params)

    # Create directory for this run
    base_dirpath = f"results_cpcc{args.use_cpcc}_cpcc_center{args.cpcc_center}"
    dirpath = f"{base_dirpath}_run_{run_id + 1}"
    
    # 确保目录存在
    if os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    os.makedirs(dirpath, exist_ok=True)
    
    # Define callbacks
    early_stop_callback = EarlyStopping(monitor="val/aupr", mode="max", patience=5, verbose=True)
    checkpoint_callback = ModelCheckpoint(monitor="val/aupr", mode="max", dirpath=dirpath, filename="best-checkpoint-all-attributes", save_top_k=1, verbose=True)
    trainer = Trainer(max_epochs=epochs,
                      logger = logger, 
                      accelerator='gpu', 
                      log_every_n_steps=1,
                      devices=torch.cuda.device_count(), 
                      strategy='ddp_find_unused_parameters_true', 
                      enable_progress_bar=True,
                      enable_model_summary=True,
                      callbacks=[checkpoint_callback, early_stop_callback],)
    
    trainer.fit(model, train_dataloader, val_dataloader)
    torch.save(model.state_dict(), f"{dirpath}/model.pth")

    # Test the model and collect results
    test_results = trainer.test(ckpt_path=f"{dirpath}/best-checkpoint-all-attributes.ckpt", 
                               dataloaders=test_dataloader)
    
    # Extract metrics from test results
    if test_results and len(test_results) > 0:
        test_metrics = test_results[0]
        print(f"Run {run_id + 1} Test Results:")
        for key, value in test_metrics.items():
            print(f"  {key}: {value:.4f}")
        
        return test_metrics
    else:
        print(f"Run {run_id + 1}: No test results available")
        return {}


def hyper_search_(args, params):
    hyperparameter_options = {
        'dropout': [0.1, 0.2, 0.3, 0.4, 0.5],
        'num_layers': [1, 2, 3, 4],
        'decay_rate': [0.01, 0.02, 0.03],
        'patient_mode': [
            "joint", 
            "graph", 
            "node"
            ]
    }
    for task in [
        "mortality", 
        "readmission", 
        "lenofstay", 
        "drugrec"
        ]:
        hyperparameter_options["task"] = [task]
        for hp_name, hp_options in hyperparameter_options.items():
            print(f"now searching for {hp_name}...")
            for hp_value in hp_options:
                print(f"now searching for {hp_name}={hp_value}...")
                params_copy = params.copy()
                params_copy[hp_name] = hp_value
                for i in range(10):
                    single_run(args, params_copy)


def main():
    args = construct_args()
    dataset, task, batch_size, hidden_dim, epochs, lr, weight_decay, dropout, num_layers, \
     decay_rate, alpha, beta, hyper_search, freeze, attn_init, in_drop_rate, kg_ratio, pos_enc_dim, debug, model_name, mimic_dir_path, save_result_path = \
        args.dataset, args.task, args.batch_size, args.hidden_dim, args.epochs, args.lr, args.weight_decay, \
            args.dropout, args.num_layers, args.decay_rate, args.alpha, args.beta, args.hyperparameter_search, args.freeze_emb, args.attn_init, args.in_drop_rate, args.kg_ratio, args.pos_enc_dim, args.debug, args.model, args.mimic_dir_path, args.save_result_path

    parameters = {
        "dataset": dataset,
        "task": task,
        "batch_size": batch_size,
        "hidden_dim": hidden_dim,
        "epochs": epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "num_layers": num_layers,
        "decay_rate": decay_rate,
        "alpha": alpha,
        "beta": beta,
        "freeze": freeze,
        "attn_init": 'False',
        "in_drop_rate": in_drop_rate,
        "kg_ratio": kg_ratio,
        "train_ratio": 1.0,
        "feat_ratio": 1.0,
        "model": model_name,
        'pos_enc_dim': pos_enc_dim,
        'debug': debug,
        'mimic_dir_path': mimic_dir_path,
        'save_result_path': save_result_path
    }
    print("run parameters: ")
    print(parameters)

    if hyper_search:
        # hyperparameter search
        print("Hyperparameter search...")
        hyper_search_(args, parameters)

    else:
        # Multiple runs for averaging results
        print(f"Starting {args.num_runs} runs for averaging results...")
        all_results = []
        
        for run_id in range(args.num_runs):
            print(f"\n{'='*60}")
            print(f"Starting Run {run_id + 1}/{args.num_runs}")
            print(f"{'='*60}")
            
            # Create logger for this run
            from pytorch_lightning.loggers import WandbLogger
            wandb_logger = WandbLogger(project="EHR_experiment",
                name = "Model_Name_{}_Batch_size_{}_Epochs_{}_Layers_{}_LR_{}_MemorySize_{}_Run_{}".format(
                    model_name, batch_size, epochs, num_layers, lr, args.memory_bank_size, run_id + 1),
                config={
                "dataset": dataset,
                "task": task,
                "model": model_name,
                "epochs": epochs,
                "lr": lr,
                "mimic_dir_path": mimic_dir_path,
                "batch_size": batch_size,
                "degree": 50,
                "hidden_dim": hidden_dim,
                "run_id": run_id + 1,
                "total_runs": args.num_runs,
                "base_seed": args.base_seed,
                "use_cpcc": args.use_cpcc,
                "cpcc_lamb": args.cpcc_lamb,
                "cpcc_distance_type": args.cpcc_distance_type,
                }
            )
            
            # Run single experiment
            run_results = single_run(args, parameters, wandb_logger, run_id)
            all_results.append(run_results)
            
            print(f"Run {run_id + 1} completed!")
        
        # Calculate and display average results
        print(f"\n{'='*60}")
        print("FINAL AVERAGED RESULTS")
        print(f"{'='*60}")
        
        if all_results:
            # Calculate averages
            avg_results = {}
            std_results = {}
            
            # Get all metric keys
            all_keys = set()
            for result in all_results:
                all_keys.update(result.keys())
            
            for key in all_keys:
                values = [result.get(key, 0) for result in all_results if key in result]
                if values:
                    avg_results[key] = np.mean(values)
                    std_results[key] = np.std(values)
            
            # Display results
            print(f"Results averaged over {args.num_runs} runs:")
            for key in sorted(avg_results.keys()):
                if key.startswith('test/'):
                    avg_val = avg_results[key]
                    std_val = std_results[key]
                    print(f"  {key}: {avg_val:.4f} ± {std_val:.4f}")
            
            # Save results to file
            results_summary = {
                'num_runs': args.num_runs,
                'base_seed': args.base_seed,
                'parameters': parameters,
                'individual_results': all_results,
                'averaged_results': avg_results,
                'std_results': std_results
            }
            
            import json
            results_file = f"results_summary_{dataset}_{task}_{model_name}_runs_{args.num_runs}.json"
            with open(results_file, 'w') as f:
                json.dump(results_summary, f, indent=2)
            
            print(f"\nDetailed results saved to: {results_file}")
        else:
            print("No results collected!")


if __name__ == '__main__':
    main()