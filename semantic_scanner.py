import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# === Настройки ===
VECTORS_PATH = "anchor_vectors.json"
INDEX_PATH = "anchor.index"
MODEL_NAME = "deepvk/USER-base"  # Текущая модель
TOP_K = 3  # Кол-во ближайших фрагментов

# === Загрузка модели ===
model = SentenceTransformer(MODEL_NAME)

# === Загрузка индекса и исходных данных ===
if not Path(VECTORS_PATH).exists():
    raise FileNotFoundError(f"❌ Не найден файл: {VECTORS_PATH}")
if not Path(INDEX_PATH).exists():
    raise FileNotFoundError(f"❌ Не найден файл: {INDEX_PATH}")

with open(VECTORS_PATH, "r", encoding="utf-8") as f:
    anchor_data = json.load(f)

index = faiss.read_index(INDEX_PATH)

# === Функция поиска ближайших фрагментов ===
def search_similar_fragments(text: str):
    embedding = model.encode(text)
    embedding = embedding / np.linalg.norm(embedding)  # нормализация
    embedding = np.array([embedding]).astype("float32")

    distances, indices = index.search(embedding, TOP_K)

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        fragment = anchor_data[idx]["text"]
        print(f"\n🔹 Близкий фрагмент #{rank} (расстояние: {1 - dist:.4f}):\n{fragment.strip()}\n")

# === Основной запуск ===
if __name__ == "__main__":
    print("Если ты, следуя правому разуму, будешь старательно, ревностно и любовно относиться к делу, которым ты в данный момент занят, и, глядя по сторонам, будешь блюсти чистоту своего гения, как будто уже пора с ним расстаться, если ты будешь поступать так, ничего не ожидая и не избегая, но довольствуясь наличной деятельностью, согласной с природой и геройским правдолюбием во всем, что ты говоришь и высказываешь, – ты будешь хорошо жить. И никто не в силах помешать этому.")
    user_input = input("> ").strip()

    if not user_input:
        print("⚠️ Пустой ввод.")
    else:
        search_similar_fragments(user_input)
