import os
import json
import faiss
import numpy as np
from generator import RussianBabelGenerator
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from uuid import uuid4

# === Настройки ===
MODEL_NAME = "deepvk/USER-base"
INDEX_PATH = "anchor.index"
VECTORS_PATH = "anchor_vectors.json"
OUTPUT_PATH = "semantic_matches.jsonl"
LOG_PATH = "semantic_matches.log"
BOOKS_TO_SCAN = 10
PAGES_PER_BOOK = 10
SIMILARITY_THRESHOLD = 0.90

# === Загрузка модели и индекса ===
model = SentenceTransformer(MODEL_NAME)
with open(VECTORS_PATH, "r", encoding="utf-8") as f:
    anchor_data = json.load(f)
index = faiss.read_index(INDEX_PATH)

# === Инициализация генератора ===
generator = RussianBabelGenerator()

# === Главная функция ===
def search_across_books():
    matches = []
    seen_hexes = set()

    with open(LOG_PATH, "w", encoding="utf-8") as log:
        for _ in tqdm(range(BOOKS_TO_SCAN), desc="📚 Книги"):
            # Уникальный hex_id
            while True:
                hex_id = uuid4().hex
                if hex_id not in seen_hexes:
                    seen_hexes.add(hex_id)
                    break

            for page_num in range(PAGES_PER_BOOK):
                text = generator.generate_page(hex_id, page_num)
                embedding = model.encode(text)
                embedding = embedding / np.linalg.norm(embedding)
                embedding = np.array([embedding]).astype("float32")

                distances, indices = index.search(embedding, 3)

                for idx, dist in zip(indices[0], distances[0]):
                    similarity = 1 - dist
                    if similarity >= SIMILARITY_THRESHOLD:
                        match = {
                            "hex_id": hex_id,
                            "page_num": page_num,
                            "similarity": round(similarity, 4),
                            "anchor_fragment": anchor_data[idx]["text"].strip(),
                            "page_text": text[:300] + ("..." if len(text) > 300 else "")
                        }
                        matches.append(match)
                        print(f"\n🔎 Совпадение!")
                        print(f"📘 Страница: hex_id={hex_id}, page_num={page_num}")
                        print(f"🧠 Сходство: {similarity:.4f}")
                        print(f"📝 Фрагмент:\n{text[:300]}\n")
                        log.write(f"MATCH {hex_id}:{page_num} → similarity {similarity:.4f}\n")

                        break  # Один матч на страницу достаточно

    if matches:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for match in matches:
                f.write(json.dumps(match, ensure_ascii=False) + "\n")
        print(f"\n✅ Найдено совпадений: {len(matches)}. Результат → {OUTPUT_PATH}")
    else:
        print("😕 Совпадений не найдено.")

if __name__ == "__main__":
    search_across_books()
