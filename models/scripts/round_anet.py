import json

raw_path = "./data/anet_gt/train.json"
with open(raw_path, 'r') as fr:
    raw_json = json.load(fr)

new_json = []
sampleid = 0
print(len(raw_json))

for i in raw_json:
    vid, duration, se, senten = i[:4]
    duration = round(duration, 2)
    new_sample = [vid, duration, se, senten]
    new_sample.append(sampleid)
    new_json.append(new_sample)
    assert i[2][1] <= i[1] , new_sample
    sampleid += 1


new_path = "./data/tmp/train.json"
with open(new_path, 'w') as fr:
    json.dump(new_json, fr)

print(len(new_json))
