import os
import json
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("deepvk/USER-base")

def split_into_chunks(text: str, min_len=100) -> list[str]:
    paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) >= min_len]
    return paragraphs

def build_anchor_from_book(input_path: str, output_path: str = "anchor_vectors.jsonl"):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Файл не найден: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        full_text = f.read()

    chunks = split_into_chunks(full_text)
    print(f"📚 Обнаружено фрагментов: {len(chunks)}")

    embeddings = model.encode(chunks, batch_size=8, show_progress_bar=True)

    results = []
    for chunk, vector in zip(chunks, embeddings):
        results.append({
            "text": chunk,
            "vector": vector.tolist()
        })

    with open(output_path, "w", encoding="utf-8") as fout:
        for entry in results:
            json.dump(entry, fout, ensure_ascii=False)
            fout.write("\n")

    print(f"✅ Сохранено векторных якорей: {len(results)} → {output_path}")


if __name__ == "__main__":
    build_anchor_from_book("mark_aurelius_naidine_s_soboi.txt")

