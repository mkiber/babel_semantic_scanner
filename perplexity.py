import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

class RussianPerplexity:
    def __init__(self, model_name="ai-forever/rugpt2large"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def calculate_perplexity(self, text: str) -> float:
        encodings = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = encodings.input_ids.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            return math.exp(loss.item())


    def sliding_window_analysis(self, text: str, window_size=1024, stride=512, threshold=50.0):
        """
        Скользящий анализ: делим текст на окна и считаем перплексию для каждого окна.
        Возвращаем фрагменты, у которых perplexity ниже порога.
        """
        results = []
        start = 0
        while start < len(text):
            chunk = text[start:start + window_size]
            if len(chunk.strip()) < 10:
                start += stride
                continue
            try:
                ppl = self.calculate_perplexity(chunk)
                if ppl <= threshold:
                    results.append({
                        "start": start,
                        "end": start + window_size,
                        "perplexity": ppl,
                        "text": chunk
                    })
            except Exception as e:
                print(f"Ошибка при расчёте perplexity: {e}")
            start += stride
        return results

if __name__ == "__main__":
    pfilter = RussianPerplexity()
    long_text = "..."  # Сюда вставь любой длинный текст

    results = pfilter.sliding_window_analysis(long_text, window_size=1024, stride=512, threshold=40.0)

    for res in results:
        print(f"\n--- Окно {res['start']}–{res['end']} ---")
        print(f"Perplexity: {res['perplexity']:.2f}")
        print(res['text'][:300], "...")

