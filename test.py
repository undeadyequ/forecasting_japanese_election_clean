import json

file = "/result/feature_import1.json"

with open(file, "r") as f:
    d1 = json.load(f)

with open(file, "w") as f:
    json.dump(d1, f, indent=4)