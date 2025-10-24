import json
import numpy as np
import faiss
import os
import argparse

parser = argparse.ArgumentParser(description="Индексирует якорные вектора из .jsonl файла")
parser.add_argument("--input", default="anchor_vectors.jsonl", help="Путь к .jsonl с эмбеддингами")
parser.add_argument("--output", default="anchor.index", help="Путь к выходному FAISS индексу")
args = parser.parse_args()

VECTORS_PATH = args.input
INDEX_PATH = args.output


def build_faiss_index():
    if not os.path.exists(VECTORS_PATH):
        raise FileNotFoundError(f"❌ Не найден файл {VECTORS_PATH}. Сначала запусти anchor_builder.py")

    with open(VECTORS_PATH, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if not data:
        raise ValueError("❌ Векторные данные пусты. Проверь входной текст в anchor_builder.py")

    # Извлекаем эмбеддинги
    embeddings = np.array([entry["vector"] for entry in data]).astype("float32")

    # Нормализуем для cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("⚠️ Найден эмбеддинг с нулевой нормой. Возможно, пустой текст.")
    normalized_embeddings = embeddings / norms

    dim = normalized_embeddings.shape[1]
    print(f"📐 Размерность эмбеддинга: {dim}")
    print(f"📦 Загружено фрагментов: {len(normalized_embeddings)}")

    # Создаём индекс на косинусное сходство
    index = faiss.IndexFlatIP(dim)
    index.add(normalized_embeddings)

    faiss.write_index(index, INDEX_PATH)
    print(f"✅ Индекс сохранён в {INDEX_PATH}")

if __name__ == "__main__":
    build_faiss_index()
