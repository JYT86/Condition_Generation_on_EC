from typing import Tuple, List, Any, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from datasets import DatasetDict, Dataset
from sklearn.model_selection import train_test_split

import pandas as pd
import numpy as np

from omegaconf import OmegaConf



def _get_brenda_data(
        file_path: str,
        if_trembl: bool = True
) -> Tuple[List[str], np.ndarray]:
    print('before read csv')
    df = pd.read_csv(file_path)
    if if_trembl == False:
        seqs = df[df['source']=='swiss-prot']['sequence'].tolist()
        ecs = df[df['source'] == 'swiss-prot']['label'].tolist()   
    else:
        seqs, ecs = df['sequence'].tolist(), df['label'].tolist()

    labels = []
    print('after read csv')
    for seq, ec in zip(seqs, ecs):
        label_split = ec.split('.')
        labels.append(label_split)
    print('after process label')

    return seqs, np.array(labels, dtype=int)


def get_datasetdict(
        name: str,
        file_path: str,
        max_len: int,
        train_ratio: float = 0.9,
        random_state: int = 42,
        if_trembl: bool = None,
        
):
    if name == 'brenda':
        assert if_trembl is not None
        seqs, labels = _get_brenda_data(file_path, if_trembl)
    print('after get data')

    train_ids, valid_ids = train_test_split(np.arange(len(seqs)), train_size=train_ratio, random_state=random_state)
    train_seqs = [seqs[i] for i in train_ids]
    train_labels = [labels[i] for i in train_ids]
    valid_seqs = [seqs[i] for i in valid_ids]
    valid_labels = [labels[i] for i in valid_ids]
    print('after split train test')

    # dataset_dict = DatasetDict({
    #     'train': Dataset.from_dict({'sequence': train_seqs, 'label': train_labels}), 
    #     'valid': Dataset.from_dict({'sequence': valid_seqs, 'label': valid_labels})
    # })

    train_dataset = Dataset.from_dict({
        'sequence': [seqs[i] for i in train_ids],
        'label': [labels[i] for i in train_ids]
    })
    
    valid_dataset = Dataset.from_dict({
        'sequence': [seqs[i] for i in valid_ids], 
        'label': [labels[i] for i in valid_ids]
    })
    
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset
    })

    def tokenization(example: Dict):
        aalists = list('*ACDEFGHIKLMNPQRSTVWY')
        aadicts = {c:i for i, c in enumerate(aalists)}
        text = example['sequence']
        
        # 处理单个序列或序列列表
        if isinstance(text, str):
            # 单个序列
            len_t = len(text)
            tokens = [aadicts[c] for c in text[:max_len] + '*' * max(0, max_len - len_t)]
            return {
                'input_ids': tokens,
                'padding_mask': [1] * min(len_t, max_len) + [0] * max(0, max_len - len_t)
            }
        else:
            # 批量序列
            batch_tokens = []
            batch_padding_mask = []
            for t in text:
                len_t = len(t)
                tokens = [aadicts[c] for c in t[:max_len] + '*' * max(0, max_len - len_t)]
                padding_mask = [1] * min(len_t, max_len) + [0] * max(0, max_len - len_t)
                
                batch_tokens.append(tokens)
                batch_padding_mask.append(padding_mask)
            return {
                'input_ids': batch_tokens, 'attention_mask': batch_padding_mask
            }
    
    print('before map')
    for split in dataset_dict:
        dataset_dict[split] = dataset_dict[split].map(
            tokenization, 
            batched=True
        )
    return dataset_dict.with_format('torch')

def get_datasetdict_chunked_creation(
        name: str,
        file_path: str,
        max_len: int,
        train_ratio: float = 0.9,
        random_state: int = 42,
        if_trembl: bool = None,
        chunk_size: int = 50000  # 每5万条创建一个chunk
):
    if name == 'brenda':
        assert if_trembl is not None
        seqs, labels = _get_brenda_data(file_path, if_trembl)
    
    train_ids, valid_ids = train_test_split(np.arange(len(seqs)), train_size=train_ratio, random_state=random_state)
    
    def create_dataset_chunked(ids, seqs, labels, split_name):
        """分块创建Dataset"""
        datasets = []
        
        for i in range(0, len(ids), chunk_size):
            chunk_ids = ids[i:i+chunk_size]
            chunk_seqs = [seqs[idx] for idx in chunk_ids]
            chunk_labels = [labels[idx] for idx in chunk_ids]
            
            # 创建小块的Dataset
            chunk_dataset = Dataset.from_dict({
                'sequence': chunk_seqs,
                'label': chunk_labels
            })
            
            # 立即进行tokenization，然后释放内存
            chunk_dataset = chunk_dataset.map(
                lambda example: tokenize_single(example, max_len),
                batched=False,
                desc=f"Tokenizing {split_name} chunk {i//chunk_size + 1}"
            )
            
            datasets.append(chunk_dataset)
            print(f"{split_name}: 已完成 {min(i+chunk_size, len(ids))}/{len(ids)}")
        
        # 合并所有chunk
        if len(datasets) > 1:
            from datasets import concatenate_datasets
            return concatenate_datasets(datasets)
        else:
            return datasets[0]
    
    def tokenize_single(example, max_len):
        aalists = list('*ACDEFGHIKLMNPQRSTVWY')
        aadicts = {c:i for i, c in enumerate(aalists)}
        
        text = example['sequence']
        len_t = len(text)
        tokens = [aadicts[c] for c in text[:max_len] + '*' * max(0, max_len - len_t)]
        padding_mask = [1] * min(len_t, max_len) + [0] * max(0, max_len - len_t)
        
        return {
            'input_ids': tokens,
            'attention_mask': padding_mask,
            'label': example['label']
        }
    
    print("创建训练集...")
    train_dataset = create_dataset_chunked(train_ids, seqs, labels, "train")
    
    print("创建验证集...")
    valid_dataset = create_dataset_chunked(valid_ids, seqs, labels, "valid")
    
    return DatasetDict({
        'train': train_dataset,
        'valid': valid_dataset
    }).with_format('torch')


def build_dataloaders(
        rank: int,
        world_size: int,
        name: str,
        file_path: str,
        max_len: int,
        distributed: bool,
        train_batch_size: int,
        valid_batch_size: int,
)-> Dict[str, DataLoader]:
    
    # ds = get_datasetdict(
    #     name,
    #     file_path,
    #     max_len,
    #     if_trembl=True
    # )

    ds = get_datasetdict_chunked_creation(
        name,
        file_path,
        max_len,
        train_ratio=0.9,
        if_trembl=True,
        chunk_size=10000,
    )

    train_sampler = DistributedSampler(ds['train'], num_replicas=world_size, rank=rank) if distributed else None
    valid_sampler = DistributedSampler(ds['valid'], num_replicas=world_size, rank=rank) if distributed else None

    train_loader = DataLoader(
        ds['train'], batch_size=train_batch_size, shuffle=(train_sampler is None), sampler=train_sampler 
    )
    valid_loader = DataLoader(
        ds['valid'], batch_size=valid_batch_size, shuffle=False, sampler=valid_sampler
    )

    loaders = {
        'train_loader': train_loader,
        'valid_loader': valid_loader,
        'train_sampler': train_sampler,
        'valid_sampler': valid_sampler
    }
    return loaders

    
    





    

    