import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from cpp_generator import cpp_generate_page
from uuid import uuid4
import time

MODEL_NAME = "deepvk/USER-base"

print("🧠 Загружаем модель...")
model = SentenceTransformer(MODEL_NAME)

def run_semantic_exploration(
    num_hex: int = 10,
    pages_per_hex: int = 10,
    similarity_threshold: float = 0.9,  # ← именно threshold, не similarity_threshold
    anchor_vector_path: str = "anchor_vectors.jsonl",
    anchor_index_path: str = "anchor.index"
):
    print("📦 Загружаем индекс и якоря...")
    index = faiss.read_index(anchor_index_path)

    with open(anchor_vector_path, "r", encoding="utf-8") as f:
        anchor_data = [json.loads(line) for line in f]

    matches = []
    start_time = time.time()

    def generate_hex_ids(n):
        return [str(uuid4())[:6] for _ in range(n)]

    for hex_id in generate_hex_ids(num_hex):
        for page_num in range(pages_per_hex):
            text = cpp_generate_page(hex_id, page_num)
            embedding = model.encode(text)

            D, I = index.search(np.array([embedding]).astype("float32"), k=1)
            similarity = float(1 - D[0][0])  # cosine similarity

            if similarity >= similarity_threshold:
                match = {
                    "hex_id": hex_id,
                    "page_num": page_num,
                    "similarity": round(similarity, 4),
                    "anchor_fragment": anchor_data[I[0][0]]["text"].strip(),
                    "page_text": text[:300] + ("..." if len(text) > 300 else "")
                }
                matches.append(match)

    elapsed = time.time() - start_time
    print(f"✅ Сканирование завершено: найдено {len(matches)} совпадений за {elapsed:.2f} сек")
    return matches

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

