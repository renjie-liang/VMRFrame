from utils.utils import load_json, save_json
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from pprint import pprint
from tqdm import tqdm

gt_path = "./data/charades_gt/train.json"
new_path = "./data/charades_SimilarSentence/train.json"
gt_data = load_json(gt_path)

sentences = []
for sample in gt_data:
# for sample, _ in zip(gt_data, range(100)):
    vid, duration, se_time, s, _ = sample
    sentences.append(s)

model = SentenceTransformer('bert-base-nli-mean-tokens')
embeddings = model.encode(sentences)
scores = cosine_similarity(embeddings, embeddings)

new_data = []
for i in tqdm(range(len(sentences))):
    vid, duration, se_time, _, _ = gt_data[i]
    for k in range(len(sentences)):
        if scores[i, k] > 0.98:
            # print(scores[i, k], sentences[k])
            new_data.append([vid, duration, se_time, sentences[k]])
save_json(new_data, new_path)
print("{} --> {}".format(len(gt_data), len(new_data)))