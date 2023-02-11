
import os
import codecs
import numpy as np
from tqdm import tqdm
from collections import Counter
from nltk.tokenize import word_tokenize
# from util.data_util import load_json, load_lines, load_pickle, save_pickle, time_to_index
from utils.utils import load_json, load_pickle, save_pickle, time_idx
import glob

PAD, UNK = "<PAD>", "<UNK>"

def process_data(data_file):
    data = load_json(data_file)
    results = []
    for record in tqdm(data, total=len(data), desc='process dataset'):
        vid, duration, (stime, etime), sentence = record[:4]
        words = word_tokenize(sentence.strip().lower(), language="english")
        tmp = {     
                'vid'       : str(vid), 
                'stime'     : stime, 
                'etime'     : etime,
                'duration'  : round(duration, 2), 
                'sentence'  : sentence, 
                'words'     : words, 
            }
        results.append(tmp)
    return results

def load_glove(glove_path):
    vocab = list()
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove vocabulary"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            vocab.append(word)
    return set(vocab)


def filter_glove_embedding(word_dict, glove_path):
    vectors = np.zeros(shape=[len(word_dict), 300], dtype=np.float32)
    with codecs.open(glove_path, mode="r", encoding="utf-8") as f:
        for line in tqdm(f, total=2196018, desc="load glove embeddings"):
            line = line.lstrip().rstrip().split(" ")
            if len(line) == 2 or len(line) != 301:
                continue
            word = line[0]
            if word in word_dict:
                vector = [float(x) for x in line[1:]]
                word_index = word_dict[word]
                vectors[word_index] = np.asarray(vector)
    return np.asarray(vectors)

def vocab_emb_gen(datasets, emb_path):
    # generate word dict and vectors
    emb_vocab = load_glove(emb_path)
    word_counter, char_counter = Counter(), Counter()
    for data in datasets:
        for record in data:
            for word in record['words']:
                word_counter[word] += 1
                for char in list(word):
                    char_counter[char] += 1
    word_vocab = list()
    for word, _ in word_counter.most_common():
        if word in emb_vocab:
            word_vocab.append(word)
    tmp_word_dict = dict([(word, index) for index, word in enumerate(word_vocab)])
    vectors = filter_glove_embedding(tmp_word_dict, emb_path)
    word_vocab = [PAD, UNK] + word_vocab
    word_dict = dict([(word, idx) for idx, word in enumerate(word_vocab)])
    # generate character dict
    char_vocab = [PAD, UNK] + [char for char, count in char_counter.most_common() if count >= 5]
    char_dict = dict([(char, idx) for idx, char in enumerate(char_vocab)])
    return word_dict, char_dict, vectors



def load_dataset(configs):
    os.makedirs(configs.paths.cache_dir, exist_ok=True)
    cache_path = os.path.join(configs.paths.cache_dir, '{}_{}.pkl'.format(configs.task,configs.suffix))
    if not os.path.exists(cache_path):
        generate_dataset(configs, cache_path)
    return load_pickle(cache_path)


def get_vfeat_len(configs):
    feature_dir = configs.paths.feature_path
    vlen_list = glob.glob(os.path.join(feature_dir, "*.npy"))
    vfeat_lens = {}
    for vpath in tqdm(vlen_list, desc="get video feature lengths"):
        tmp = os.path.split(vpath)
        vid = tmp[-1][:-4]
        vfeat_lens[vid] = np.load(vpath).shape[0]
        # vfeat_lens[vid] = min(configs.model.vlen, np.load(vpath).shape[0])
    return vfeat_lens 


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_tlen, scope):
    dataset = list()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        # s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration']) ### ???? replace???
        # s_ind, e_ind = time_idx([record['s_time'], record['e_time']], record['duration'], vfeat_lens[vid])
        # if e_ind > vfeat_lens[vid]:
        #     print(record)
        if record['etime'] > record['duration']:
            print(record)
            record['etime'] = record['duration']

        sfrac, efrac = record['stime'] / record['duration'], record['etime'] / record['duration']
        assert 0.0 <= sfrac <= 1.0, record
        assert 0.0 <= efrac <= 1.0, record

        # bert_input = tokenizer(" ".join(record['words']), padding='max_length', 
        #                max_length = max_tlen,  truncation=True, return_tensors="pt")
        # bert_id = bert_input["input_ids"]
        # bert_mask = bert_input["attention_mask"]

        word_ids, char_ids = [], []
        for word in record['words'][0:max_tlen]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {
                # 'sample_id': record['sample_id'], 
                'vid': record['vid'], 
                'se_time': [record['stime'], record['etime']],
                'duration': record['duration'], 
                'se_frac': [sfrac, efrac], 
                # 's_ind': int(s_ind), 
                # 'e_ind': int(e_ind), 
                # 'v_len': vfeat_lens[vid], 
                'sentence': record['sentence'],
                'words': record['words'],
                'wids': word_ids,
                'cids': char_ids,
                # 'bert_id':bert_id,
                # "bert_mask":bert_mask,
                }
        dataset.append(result)
    return dataset


def generate_dataset(configs, cache_path):
    vfeat_lens = get_vfeat_len(configs)
    train_data = process_data(configs.paths.train_path)
    test_data = process_data(configs.paths.test_path)
    if configs.paths.val_path == '':
        data_list = [train_data, test_data]
    else:
        val_data = process_data(configs.paths.val_path)
        data_list = [train_data, val_data, test_data]

    # generate dataset
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, configs.paths.glove_path)
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.model.tlen, 'train')
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.model.tlen, 'test')
    if configs.paths.val_path == '':
        val_set = None
        n_val = 0 
    else:
        val_set = dataset_gen(val_data, vfeat_lens, word_dict, char_dict, configs.model.tlen, 'val')
        n_val = len(val_set)

    # save dataset
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, cache_path)
    return dataset