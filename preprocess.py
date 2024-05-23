import os
import argparse
import yaml
from phonemize import phonemize
import phonemizer
from transformers import TransfoXLTokenizer
from datasets import load_dataset, concatenate_datasets, load_from_disk
#from pebble import ProcessPool
#from concurrent.futures import TimeoutError
from tqdm import tqdm
import pickle
from dataloader import build_dataloader, FilePathDataset
from tqdm import tqdm

from concurrent.futures import ProcessPoolExecutor

def parse_args():
    parser = argparse.ArgumentParser(description="Process the Wikipedia dataset.")
    parser.add_argument("--config_path", type=str, default="Configs/config.yml", help="Path to the config file.")
    #parser.add_argument("--root_directory", type=str, default="./multilingual-phonemes", help="Root directory for processed data.")
    parser.add_argument("--root_directory", type=str, default="./wiki_phoneme", help="Root directory for processed data.")
    parser.add_argument("--num_shards", type=int, default=1000, help="Number of shards to split the dataset.")
    parser.add_argument("--num_workers", type=int, default=-1, help="Workers on CPU cores")
    return parser.parse_args()

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def initialize_components(config):
    global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True, with_stress=True)
    tokenizer = TransfoXLTokenizer.from_pretrained(config['dataset_params']['tokenizer'])
    #dataset = load_dataset("styletts2-community/multilingual-phonemes-10k-alpha", "sv")['train']
    dataset = load_dataset("wikipedia", "20220301.en")['train']
    return global_phonemizer, tokenizer, dataset

def process_shard(i, args, dataset, global_phonemizer, tokenizer):
    directory = os.path.join(args.root_directory, "shard_" + str(i))
    if os.path.exists(directory):
        print(f"Shard {i} already exists!")
        return
    print(f'Processing shard {i} ...')
    shard = dataset.shard(num_shards=args.num_shards, index=i)
    processed_dataset = shard.map(lambda t: phonemize(t['text'], global_phonemizer, tokenizer), remove_columns=['text'])
    os.makedirs(directory, exist_ok=True)
    processed_dataset.save_to_disk(directory)
'''
def process_dataset(args, dataset, global_phonemizer, tokenizer):
    #max_workers = 1
    #with ProcessPool(max_workers=max_workers) as pool:
    #    pool.map(process_shard, range(args.num_shards), args=(args, dataset, global_phonemizer, tokenizer), timeout=60)
    for i in tqdm(range(args.num_shards)):
        process_shard(i, args, dataset, global_phonemizer, tokenizer)
'''
def process_dataset(args, dataset, global_phonemizer, tokenizer):
    max_workers = os.cpu_count()  if args.num_workers < 0 else args.num_workers
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_shard, i, args, dataset, global_phonemizer, tokenizer) for i in range(args.num_shards)]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            future.result()  # Isso garante que qualquer exceção nos shards seja levantada aqui

def combine_shards(args, config):
    output = [dI for dI in os.listdir(args.root_directory) if os.path.isdir(os.path.join(args.root_directory, dI))]
    datasets = []
    for o in output:
        directory = os.path.join(args.root_directory, o)
        try:
            shard = load_from_disk(directory)
            datasets.append(shard)
            print(f"{o} loaded")
        except Exception as e:
            print(f"Failed to load {o}: {str(e)}")
            continue
    dataset = concatenate_datasets(datasets)
    dataset.save_to_disk(config['data_folder'])
    print(f'Dataset saved to {config["data_folder"]}')
    return True

if __name__ == "__main__":
    args = parse_args()
    print("Loading config...")
    config = load_config(args.config_path)
    print("Initialize components...")
    global_phonemizer, tokenizer, dataset = initialize_components(config)
    print("Processing dataset...")
    process_dataset(args, dataset, global_phonemizer, tokenizer)
    print("Combining shards...")
    final_dataset = combine_shards(args, config)
    # Further processing can be done here
