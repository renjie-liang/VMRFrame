import json

raw_path = "./data/charades_gt/test.json"
with open(raw_path, 'r') as fr:
    raw_json = json.load(fr)

new_json = []
sampleid = 0

print(len(raw_json))

for i in raw_json:
    new_sample = i[:4]
    new_sample.append(sampleid)
    new_json.append(new_sample)

    assert i[1] >= i[2][1], "{} {}".format( i[1], i[2][1])
    sampleid += 1

new_path = "./data/charades_clean/test.json"
with open(new_path, 'w') as fr:
    json.dump(new_json, fr)

print(len(new_json))
