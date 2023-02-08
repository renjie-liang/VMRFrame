
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

class CharadesProcessor:
    def __init__(self):
        super(CharadesProcessor, self).__init__()
        # self.idx_counter = 0
        pass

    # def reset_idx_counter(self):
    #     # self.idx_counter = 0
    #     pass

    def process_data(self, data):
        results = []
        for record in tqdm(data, total=len(data), desc='process charades_active'):
            vid, duration, (start_time, end_time), sentence = record[:4]
            if len(record) > 4:
                active_weight = record[-1]
            else:
                active_weight = None

            # start_time, end_time = gt_label
            words = word_tokenize(sentence.strip().lower(), language="english")

            tmp = {     
                    # 'sample_id' : self.idx_counter,
                    'vid'       : str(vid), 
                    's_time'    : start_time, 
                    'e_time'    : end_time,
                    'duration'  : duration, 
                    'words'     : words, 
                    'active_weight' :active_weight
                }
            results.append(tmp)
            # self.idx_counter += 1
        return results

    def convert(self, data_file):
        # load raw data
        data_json = load_json(data_file)
        data_collect = self.process_data(data_json)
        return data_collect



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
    cache_path = os.path.join(configs.paths.cache_dir, '{}_{}_{}.pkl'.format(configs.task, configs.model.vlen,configs.suffix))
    if not os.path.exists(cache_path):
        generate_dataset(configs, cache_path)
        # dataset = replace_data(dataset, data_dir, cache_path, configs.task)
        # return dataset
    return load_pickle(cache_path)


def get_vfeat_len(configs):
    feature_dir = configs.paths.feature_path
    vlen_list = glob.glob(os.path.join(feature_dir, "*.npy"))
    vfeat_lens = {}
    for vpath in tqdm(vlen_list, desc="get video feature lengths"):
        tmp = os.path.split(vpath)
        vid = tmp[-1][:-4]
        ll = np.load(vpath).shape[0]
        vfeat_lens[vid] = min(configs.model.vlen, ll)
    return vfeat_lens 


def dataset_gen(data, vfeat_lens, word_dict, char_dict, max_pos_len, scope):
    dataset = list()
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    for record in tqdm(data, total=len(data), desc='process {} data'.format(scope)):
        vid = record['vid']
        if vid not in vfeat_lens:
            continue
        # s_ind, e_ind, _ = time_to_index(record['s_time'], record['e_time'], vfeat_lens[vid], record['duration']) ### ???? replace???
        s_ind, e_ind = time_idx([record['s_time'], record['e_time']], record['duration'], vfeat_lens[vid])
        if e_ind > vfeat_lens[vid]:
            print(record)

        bert_input = tokenizer(" ".join(record['words']), padding='max_length', 
                       max_length = max_pos_len, 
                       truncation=True,
                       return_tensors="pt")
        bert_id = bert_input["input_ids"]
        bert_mask = bert_input["attention_mask"]


        word_ids, char_ids = [], []
        for word in record['words'][0:max_pos_len]:
            word_id = word_dict[word] if word in word_dict else word_dict[UNK]
            char_id = [char_dict[char] if char in char_dict else char_dict[UNK] for char in word]
            word_ids.append(word_id)
            char_ids.append(char_id)
        result = {
                # 'sample_id': record['sample_id'], 
                'vid': record['vid'], 
                's_time': record['s_time'],
                'e_time': record['e_time'], 
                'duration': record['duration'], 
                'words': record['words'],
                's_ind': int(s_ind), 
                'e_ind': int(e_ind), 
                'v_len': vfeat_lens[vid], 
                'w_ids': word_ids,
                'c_ids': char_ids,
                'bert_id':bert_id,
                "bert_mask":bert_mask,
                }
        dataset.append(result)
    return dataset


def generate_dataset(configs, cache_path):
    vfeat_lens = get_vfeat_len(configs)
    processor = CharadesProcessor()
    train_data = processor.convert(configs.paths.train_path)
    test_data = processor.convert(configs.paths.test_path)
    if configs.paths.val_path == '':
        data_list = [train_data, test_data]
    else:
        val_data = processor.convert(configs.paths.val_path)
        data_list = [train_data, val_data, test_data]

    # generate dataset
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, configs.paths.glove_path)
    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'train') # ???? active
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'test')
    if configs.paths.val_path == '':
        val_set = None
        n_val = 0 
    else:
        val_set = dataset_gen(val_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'val')
        n_val = len(val_set)

    # save dataset
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, cache_path)
    return dataset



def generate_dataset_BAN(configs, cache_path):
    vfeat_lens = get_vfeat_len(configs)
    # data_dir = os.path.join('data', 'dataset', configs.task + "_" + configs.suffix)
    # load data
    processor = CharadesProcessor()
    # train_data, val_data, test_data = processor.convert(data_dir)

    train_data = processor.convert(configs.paths.train_path)
    test_data = processor.convert(configs.paths.test_path)

    if configs.paths.val_path == '':
        data_list = [train_data, test_data]
    else:
        val_data = processor.convert(configs.paths.val_path)
        data_list = [train_data, val_data, test_data]

    # generate dataset
    word_dict, char_dict, vectors = vocab_emb_gen(data_list, configs.paths.glove_path)

    train_set = dataset_gen(train_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'train') # ???? active
    test_set = dataset_gen(test_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'test')
    if configs.paths.val_path == '':
        val_set = None
        n_val = 0 
    else:
        val_set = dataset_gen(val_data, vfeat_lens, word_dict, char_dict, configs.model.vlen, 'val')
        n_val = len(val_set)

    # save dataset
    dataset = {'train_set': train_set, 'val_set': val_set, 'test_set': test_set, 'word_dict': word_dict,
               'char_dict': char_dict, 'word_vector': vectors, 'n_train': len(train_set), 'n_val': n_val,
               'n_test': len(test_set), 'n_words': len(word_dict), 'n_chars': len(char_dict)}
    save_pickle(dataset, cache_path)
    return dataset