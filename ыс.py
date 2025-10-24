import json

with open("anchor_vectors.json", "r", encoding="utf-8") as f_in:
    data = json.load(f_in)

with open("anchor_vectors.jsonl", "w", encoding="utf-8") as f_out:
    for entry in data:
        json.dump(entry, f_out, ensure_ascii=False)
        f_out.write("\n")

print("✅ Конвертация завершена: anchor_vectors.jsonl создан.")
