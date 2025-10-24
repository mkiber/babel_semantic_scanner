from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import math
import requests
import json

def fetch_pages_from_api(api_url, params=None):
    """
    Generator: fetches pages from the Babylon Library API (or any paginated source).
    Yields raw text pages.
    """
    page = 1
    while True:
        query = params.copy() if params else {}
        query.update({'page': page})
        resp = requests.get(api_url, params=query)
        if resp.status_code != 200:
            break
        data = resp.json()
        texts = data.get('pages', [])
        if not texts:
            break
        for entry in texts:
            yield entry.get('text', '')
        page += 1

class PerplexityFilter:
    def __init__(self, model_name="gpt2", device=None):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

    def calculate_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=1024)
        input_ids = inputs["input_ids"].to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, labels=input_ids)
            loss = outputs.loss
            return math.exp(loss.item())

    def filter_stream(self, pages, window_size=1024, stride=512, threshold=60.0):
        """
        Streams through pages (iterable of raw texts), applies sliding window,
        computes perplexity, and yields candidates below threshold.
        """
        for text in pages:
            start = 0
            length = len(text)
            while start < length:
                chunk = text[start:start + window_size]
                if len(chunk.strip()) < 10:
                    start += stride
                    continue
                try:
                    ppl = self.calculate_perplexity(chunk)
                    if ppl <= threshold:
                        yield {
                            "start": start,
                            "end": start + window_size,
                            "perplexity": ppl,
                            "text": chunk
                        }
                except Exception as e:
                    print(f"Error calculating perplexity at {start}: {e}")
                start += stride

if __name__ == '__main__':
    API_URL = 'https://api.babylon-library.org/pages'
    # Example params, adjust according to real API:
    PARAMS = {'page_size': 10}

    # Initialize filter
    pf = PerplexityFilter(model_name='gpt2', device=None)

    # Stream pages and process
    results = []
    for candidate in pf.filter_stream(fetch_pages_from_api(API_URL, PARAMS),
                                      window_size=1024,
                                      stride=512,
                                      threshold=50.0):
        print(f"Found candidate at {candidate['start']}-{candidate['end']} with PPL={candidate['perplexity']:.2f}")
        results.append(candidate)

    # Save results to JSONL
    with open('filtered_candidates.jsonl', 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
