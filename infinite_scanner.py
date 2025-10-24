# infinite_scanner.py
import time
import json
import os
from datetime import datetime
from semantic_engine import run_semantic_exploration

OUTPUT_DIR = "semantic_results"
LOG_PATH = "scanner.log"
THRESHOLD = 0.95
HEX_BATCH = 50
PAGES_PER_HEX = 10
SLEEP_TIME = 5  # сек между циклами

os.makedirs(OUTPUT_DIR, exist_ok=True)

def log(message: str):
    """Пишет строку в лог и печатает её на экран"""
    timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{timestamp} {message}"
    print(line)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")

log("🚀 Запуск бесконечного смыслового сканирования...")

while True:
    start = time.time()
    try:
        matches = run_semantic_exploration(
            num_hex=HEX_BATCH,
            pages_per_hex=PAGES_PER_HEX,
            similarity_threshold=THRESHOLD
        )

        if matches:
            filename = f"matches_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            filepath = os.path.join(OUTPUT_DIR, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(matches, f, ensure_ascii=False, indent=2)

            log(f"💾 Найдено {len(matches)} совпадений → {filepath}")
        else:
            log("😴 Совпадений не найдено в этом цикле.")

    except Exception as e:
        log(f"❌ Ошибка: {e}")

    elapsed = time.time() - start
    log(f"⏱️ Цикл завершён за {elapsed:.2f} сек.\n")
    time.sleep(SLEEP_TIME)
