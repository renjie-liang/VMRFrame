import json

raw_path = "data/charades_reset/test.json"
with open(raw_path, 'r') as fr:
    raw_json = json.load(fr)
new_json = []
for i in raw_json:
    new_json.append(i[:4])
    assert i[1] >= i[2][1], "{} {}".format( i[1], i[2][1])


new_path = "data/charades_clean/test.json"
with open(new_path, 'w') as fr:
    json.dump(new_json, fr)