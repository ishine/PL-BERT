import os
import re
import yaml
from datasets import load_dataset, Dataset
from transformers import BertTokenizer, BertModel
from pypinyin import pinyin, lazy_pinyin, Style
import concurrent.futures
from pypinyin_dict.phrase_pinyin_data import cc_cedict
from hanziconv import HanziConv
import phonemizer
cc_cedict.load()

config_path = "Configs/config.yml" # you can change it to anything else
config = yaml.safe_load(open(config_path))

tokenizer = BertTokenizer.from_pretrained(config['dataset_params']['tokenizer']) # you can use any other tokenizers if you want to

# dataset = load_dataset("wikipedia", language="zh", date="20240720", trust_remote_code=True)['train']
dataset = load_dataset("erhwenkuo/wikipedia-zhtw", cache_dir="/home/twsgxyc199/yuxianglin/backup/hf_dataset")['train']
# dataset = dataset[:10]
root_directory = "./wiki_phoneme" # set up root directory for multiprocessor processing
if not os.path.exists(root_directory):
    os.makedirs(root_directory)

import itertools
import re

from pypinyin import Style, lazy_pinyin, pinyin


def split_by_language(text):
    # Define a regex pattern to capture different scripts
    pattern = r'[\u4e00-\u9fff]+|[a-zA-Z]+(?:[-\'][a-zA-Z]+)*|\d+|[^\w\s]'

    # Find all matches in the text
    matches = re.findall(pattern, text)

    # Group continuous segments of the same language
    result = []
    current_segment = matches[0]

    last_is_english = False
    for match in matches[1:]:
        if re.match(r'[\u4e00-\u9fff]', current_segment[-1]) and re.match(r'[\u4e00-\u9fff]', match):
            current_segment += match
            last_is_english = False
        elif re.match(r'[a-zA-Z]', current_segment[-1]) and re.match(r'[a-zA-Z]', match):

            current_segment += " " + match
        elif re.match(r'\d', current_segment[-1]) and re.match(r'\d', match):
            current_segment += match
            last_is_english = False
        else:
            result.append(current_segment)
            current_segment = match
            last_is_english = False

    result.append(current_segment)

    return result


def num2chinese(num, big=False, simp=True, o=False, twoalt=False):
    """
    Converts numbers to Chinese representations.

    `big`   : use financial characters.
    `simp`  : use simplified characters instead of traditional characters.
    `o`     : use 〇 for zero.
    `twoalt`: use 两/兩 for two when appropriate.

    Note that `o` and `twoalt` is ignored when `big` is used, 
    and `twoalt` is ignored when `o` is used for formal representations.
    """
    # check num first
    if num in ["，", "。"]:
        return num

    nd = str(num)
    try:
        if abs(float(nd)) >= 1e48:
            raise ValueError('number out of range')
        elif 'e' in nd:
            raise ValueError('scientific notation is not supported')
    except:
        pass
    c_symbol = '正负点' if simp else '正負點'
    if o:  # formal
        twoalt = False
    if big:
        c_basic = '零壹贰叁肆伍陆柒捌玖' if simp else '零壹貳參肆伍陸柒捌玖'
        c_unit1 = '拾佰仟'
        c_twoalt = '贰' if simp else '貳'
    else:
        c_basic = '〇一二三四五六七八九' if o else '零一二三四五六七八九'
        c_unit1 = '十百千'
        if twoalt:
            c_twoalt = '两' if simp else '兩'
        else:
            c_twoalt = '二'
    c_unit2 = '万亿兆京垓秭穰沟涧正载' if simp else '萬億兆京垓秭穰溝澗正載'
    revuniq = lambda l: ''.join(k for k, g in itertools.groupby(reversed(l)))
    nd = str(num)
    result = []
    if nd[0] == '+':
        result.append(c_symbol[0])
    elif nd[0] == '-':
        result.append(c_symbol[1])
    if '.' in nd:
        integer, remainder = nd.lstrip('+-').split('.')
    else:
        integer, remainder = nd.lstrip('+-'), None
    if int(integer):
        splitted = [integer[max(i - 4, 0):i]
                    for i in range(len(integer), 0, -4)]
        intresult = []
        for nu, unit in enumerate(splitted):
            # special cases
            if int(unit) == 0:  # 0000
                intresult.append(c_basic[0])
                continue
            elif nu > 0 and int(unit) == 2:  # 0002
                intresult.append(c_twoalt + c_unit2[nu - 1])
                continue
            ulist = []
            unit = unit.zfill(4)
            for nc, ch in enumerate(reversed(unit)):
                if ch == '0':
                    if ulist:  # ???0
                        ulist.append(c_basic[0])
                elif nc == 0:
                    ulist.append(c_basic[int(ch)])
                elif nc == 1 and ch == '1' and unit[1] == '0':
                    # special case for tens
                    # edit the 'elif' if you don't like
                    # 十四, 三千零十四, 三千三百一十四
                    ulist.append(c_unit1[0])
                elif nc > 1 and ch == '2':
                    ulist.append(c_twoalt + c_unit1[nc - 1])
                else:
                    ulist.append(c_basic[int(ch)] + c_unit1[nc - 1])
            ustr = revuniq(ulist)
            if nu == 0:
                intresult.append(ustr)
            else:
                intresult.append(ustr + c_unit2[nu - 1])
        result.append(revuniq(intresult).strip(c_basic[0]))
    else:
        result.append(c_basic[0])
    if remainder:
        result.append(c_symbol[2])
        result.append(''.join(c_basic[int(ch)] for ch in remainder))
    return ''.join(result)

def fix_number(s):
    s = num2chinese(s)
    return lazy_pinyin(s)

def traditional_to_simplified(traditional_text):
	"""Convert Traditional Chinese to Simplified Chinese."""
	simplified_text = HanziConv.toSimplified(traditional_text)
	# simplified_text = remove_punctuation(simplified_text)
	return simplified_text

def text_to_phonemes(text, global_phonemizer):
	"""Convert Chinese text to phonemes using phonemizer."""
	
	# Rule-based
	text = text.replace(".", "點")

	segments = split_by_language(text)
	phonemes = []

	for segment in segments:
		if re.match(r'[a-zA-Z]', segment): 
			phonemes += global_phonemizer.phonemize([segment])
		elif re.match(r'[\u4e00-\u9fff]', segment):
			phonemes += lazy_pinyin(segment, style=Style.TONE3, neutral_tone_with_five=True, errors=fix_number)
		elif re.match(r'\d', segment):
			phonemes += fix_number(segment)
		else: 
			phonemes += segment

	phonemes = " ".join(phonemes)
	return phonemes

def text_normalize(text):
    # Regular expression to match digits and non-Chinese characters
    pattern = r'[^\u4e00-\u9fff。 ， ； ： ？ ！ …… 、 “ ” ‘ ’ 「 」 『 』 （ ） ［ ］ 《 》 { } —— ～ · ]'
    
    # Replace matched characters with an empty string
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text
    
_chinese_punctuation = "。 ， ； ： ？ ！ …… 、 “ ” ‘ ’ 「 」 『 』 （ ） ［ ］ 《 》 { } —— ～ · "
global_phonemizer = phonemizer.backend.EspeakBackend(language='en-us', preserve_punctuation=True)
def phonemize(text, tokenizer):
    """Convert Chinese text to phonemes using phonemizer."""
    # print(text)
    # print()
    # text = split_by_language(text)
    words = tokenizer.tokenize(text)

    input_ids = []
    phonemes = []
    for w in words: 
        if re.match(r'[a-zA-Z]', w): 
            phonemes.append(global_phonemizer.phonemize([w]))
        elif re.match(r'[\u4e00-\u9fff]', w):
            phonemes.append(lazy_pinyin(w, style=Style.TONE3, neutral_tone_with_five=True, errors=fix_number))
        elif re.match(r'\d', w):
            phonemes.append(fix_number(w))
        else: 
            phonemes.append(w)  
        input_ids.append(tokenizer.encode(w)[1:-1])
    
    # phonemes_bad = [lazy_pinyin(word, style=Style.TAIWAN, neutral_tone_with_five=True)[0] if word not in _chinese_punctuation else word for word in words]
    # phonemes_bad = [lazy_pinyin(word, style=Style.TONE3, neutral_tone_with_five=True)[0] if word not in _chinese_punctuation else word for word in words]

    # input_ids = [[tokenizer.encode(word)[1:-1]] for word in words]
    # phonemes = [p for p in phonemes_bad]
    
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
# dataset = Dataset.from_dict(dataset)
processed_dataset = dataset.map(lambda t: phonemize(t['text'], tokenizer), batched=False, num_proc=128)
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

