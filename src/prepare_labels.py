import json

real_json = "data/real_cifake_preds.json"
fake_json = "data/fake_cifake_preds.json"
output_json = "data/train_labels.json"

with open(real_json) as f:
    real_data = json.load(f)
with open(fake_json) as f:
    fake_data = json.load(f)

combined = []
for r in real_data:
    combined.append({"index": r["index"], "prediction": "real"})
for f in fake_data:
    combined.append({"index": f["index"], "prediction": "fake"})

with open(output_json, "w") as f:
    json.dump(combined, f, indent=4)

print("✅ Combined train_labels.json created successfully.")
