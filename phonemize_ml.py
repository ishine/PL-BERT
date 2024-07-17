'''
Source: https://huggingface.co/styletts2-community/data-preprocessing-scripts-moved-to-github/blob/main/process_lang.py
'''
import string
from nltk.tokenize import TweetTokenizer
nltk_tokenizer = TweetTokenizer()
from nltk.tokenize.treebank import TreebankWordDetokenizer
detokenizer = TreebankWordDetokenizer()

def word_tokenize(text): return nltk_tokenizer.tokenize(text)
def detokenize(tokens): return detokenizer.detokenize(tokens)

def generate_trigrams(tokens):
    trigrams = []
    for i in range(len(tokens) - 2):
        trigram = tokens[i:i + 3]
        trigrams.append(trigram)
    return trigrams

def phonemize(text, global_phonemizer, tokenizer):
    words = word_tokenize(text)
    trigrams = generate_trigrams(words)
    if len(trigrams) == 0:
        print("Empty trigram...")      
        return {'input_ids' : None, 'phonemes': None}
    pairs = []
    
    k = trigrams[0]
    trigram = detokenize(k)
    phonemes = word_tokenize(global_phonemizer.phonemize([trigram], strip=True)[0])

    word = k[0]

    if len(phonemes) == 3:
        pairs.append((tokenizer.encode(word)[1:-1], phonemes[0]))
    else:
        pairs.append((tokenizer.encode(word)[1:-1], global_phonemizer.phonemize([k[0]], strip=True)[0]))

    for k in trigrams:
        trigram = detokenize(k)
        word = k[1]
        if k[1] in string.punctuation:
            pairs.append((tokenizer.encode(word)[1:-1], word))
            continue
        phonemes = word_tokenize(global_phonemizer.phonemize([trigram], strip=True)[0])

        if len(phonemes) == 3:
            pairs.append((tokenizer.encode(word)[1:-1], phonemes[1]))
        else:
            pairs.append((tokenizer.encode(word)[1:-1], global_phonemizer.phonemize([k[1]], strip=True)[0]))

    k = trigrams[-1]
    trigram = detokenize(k)
    phonemes = word_tokenize(global_phonemizer.phonemize([trigram], strip=True)[0])

    word = k[-1]
    if len(phonemes) == 3:
        pairs.append((tokenizer.encode(word)[1:-1], phonemes[-1]))
    else:
        pairs.append((tokenizer.encode(word)[1:-1], global_phonemizer.phonemize([k[-1]], strip=True)[0]))


    input_ids = []
    phonemes = []

    for p in pairs:
        input_ids.append(p[0])
        phonemes.append(p[1])
    assert len(input_ids) == len(phonemes)   
    return {'input_ids' : input_ids, 'phonemes': phonemes}
