import json
import faiss
import random
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from cpp_generator import cpp_generate_page  # подключаем C++ генератор

# === Конфигурация ===
MODEL_NAME = "deepvk/USER-base"
INDEX_PATH = "anchor.index"
VECTORS_PATH = "anchor_vectors.json"
RESULTS_PATH = Path("semantic_hits.jsonl")

HEX_SPACE = "abcdefghijklmnopqrstuvwxyz0123456789"
MAX_PAGES = 100
SIMILARITY_THRESHOLD = 0.90

# === Загрузка ===
model = SentenceTransformer(MODEL_NAME)
with open(VECTORS_PATH, "r", encoding="utf-8") as f:
    anchors = json.load(f)

index = faiss.read_index(INDEX_PATH)

def generate_random_hex_id(length=6):
    return ''.join(random.choices(HEX_SPACE, k=length))

def normalize(v):
    return v / np.linalg.norm(v)

def search_similar(text: str, top_k=3):
    emb = model.encode(text)
    emb = normalize(emb).astype("float32").reshape(1, -1)
    distances, indices = index.search(emb, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        similarity = 1 - dist
        if similarity >= SIMILARITY_THRESHOLD:
            results.append({
                "similarity": round(similarity, 4),
                "fragment": anchors[idx]["text"]
            })
    return results

# === Основной цикл ===
hits = []
for _ in range(MAX_PAGES):
    hex_id = generate_random_hex_id()
    page_num = random.randint(0, 1000)

    try:
        text = cpp_generate_page(hex_id, page_num)
        matches = search_similar(text, top_k=3)

        if matches:
            print(f"📌 Найдены совпадения для {hex_id}:{page_num}")
            hits.append({
                "hex_id": hex_id,
                "page_num": page_num,
                "matches": matches,
                "text": text[:500]  # первые 500 символов
            })
    except Exception as e:
        print(f"❌ Ошибка на {hex_id}:{page_num}: {e}")

# === Сохранение ===
if hits:
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        for hit in hits:
            f.write(json.dumps(hit, ensure_ascii=False) + "\n")

    print(f"\n✅ Сохранено: {len(hits)} совпадений → {RESULTS_PATH}")
else:
    print("😕 Совпадений не найдено.")
