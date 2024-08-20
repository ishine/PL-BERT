import os
import re
import yaml
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertModel
from pypinyin import pinyin, lazy_pinyin, Style
import concurrent.futures

config_path = "Configs/config.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))

tokenizer = BertTokenizer.from_pretrained(config['dataset_params']['tokenizer']) # you can use any other tokenizers if you want to

dataset = load_dataset("wikipedia", language="zh", date="20240720", trust_remote_code=True)['train']
dataset = dataset[:10]
root_directory = "./wiki_phoneme" # set up root directory for multiprocessor processing
if not os.path.exists(root_directory):
    os.makedirs(root_directory)

def text_normalize(text):
    # Regular expression to match digits and non-Chinese characters
    pattern = r'[^\u4e00-\u9fff。 ， ； ： ？ ！ …… 、 “ ” ‘ ’ 「 」 『 』 （ ） ［ ］ 《 》 { } —— ～ · ]'
    
    # Replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text
    
_chinese_punctuation = "。 ， ； ： ？ ！ …… 、 “ ” ‘ ’ 「 」 『 』 （ ） ［ ］ 《 》 { } —— ～ · "

def phonemize(text, tokenizer):
    """Convert Chinese text to phonemes using phonemizer."""
    # print(text)
    # print()
    words = tokenizer.tokenize(text)
    phonemes_bad = [lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[0] if word not in _chinese_punctuation else word for word in words]

    input_ids = [tokenizer.encode(word) for word in words]
    phonemes = [p for p in phonemes_bad]
    
    assert len(input_ids) == len(phonemes)
    return {'input_ids' : input_ids, 'phonemes': phonemes}


# num_shards = 1

# def process_shard(i):
#     directory = root_directory + "/shard_" + str(i)
#     if os.path.exists(directory):
#         print("Shard %d already exists!" % i)
#         return
#     print('Processing shard %d ...' % i)
#     shard = dataset.shard(num_shards=num_shards, index=i)
#     print('Shard %d loaded' % i)
#     processed_dataset = shard.map(lambda t: phonemize(t['text'], tokenizer), remove_columns=['text'])
#     print('Shard %d processed' % i)
#     if not os.path.exists(directory):
#         os.makedirs(directory)
#     print('Shard %d saved' % i)
#     processed_dataset.save_to_disk(directory)

# from pebble import ProcessPool
# from concurrent.futures import TimeoutError

# max_workers = 16 # change this to the number of CPU cores your machine has 

# with ProcessPool(max_workers=max_workers) as pool:
#     pool.map(process_shard, range(num_shards), timeout=60)

# from datasets import load_from_disk, concatenate_datasets

# output = [dI for dI in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory,dI))]
# datasets = []
# for o in output:
#     directory = root_directory + "/" + o
#     try:
#         shard = load_from_disk(directory)
#         datasets.append(shard)
#         print("%s loaded" % o)
#     except:
#         continue

# dataset = concatenate_datasets(datasets)
dataset = Dataset.from_dict(dataset)
processed_dataset = dataset.map(lambda t: phonemize(t['text'], tokenizer), batched=False, num_proc=24)
# dataset.save_to_disk(config['data_folder'])
print('Dataset saved to %s' % config['data_folder'])
processed_dataset.push_to_hub("Evan-Lin/wiki-phoneme", private=False)

# from simple_loader import FilePathDataset, build_dataloader

# file_data = FilePathDataset(dataset)
# loader = build_dataloader(file_data, num_workers=16, batch_size=128)

# special_token = config['dataset_params']['word_separator']

# get all unique tokens in the entire dataset

# from tqdm import tqdm

# unique_index = [special_token]
# for _, batch in enumerate(tqdm(loader)):
#     unique_index.extend(batch)
#     unique_index = list(set(unique_index))
    
# # get each token's lower case

# lower_tokens = []
# for t in tqdm(unique_index):
#     word = tokenizer.decode([t])
#     if word.lower() != word:
#         t = tokenizer.encode([word.lower()])[0]
#         lower_tokens.append(t)
#     else:
#         lower_tokens.append(t)

# lower_tokens = (list(set(lower_tokens)))

# redo the mapping for lower number of tokens

# token_maps = {}
# for t in tqdm(unique_index):
#     word = tokenizer.decode([t])
#     word = word.lower()
#     new_t = tokenizer.encode([word.lower()])[0]
#     token_maps[t] = {'word': word, 'token': unique_index.index(new_t)}


# import pickle
# with open(config['dataset_params']['token_maps'], 'wb') as handle:
#     pickle.dump(token_maps, handle)
# print('Token mapper saved to %s' % config['dataset_params']['token_maps'])

# from dataloader import build_dataloader

# train_loader = build_dataloader(dataset, batch_size=32, num_workers=0, dataset_config=config['dataset_params'])

# _, (words, labels, phonemes, input_lengths, masked_indices) = next(enumerate(train_loader))

