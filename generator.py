import hashlib

class RussianBabelGenerator:
    def __init__(self):
        self.alphabet = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя .,")
        self.alphabet_size = len(self.alphabet)
        self.page_length = 3200

        self.modulus = 2**32
        self.a = 1664525
        self.c = 1013904223

    def _get_seed(self, hex_id: str, page_num: int) -> int:
        combined = f"{hex_id}-{page_num}"
        hash_digest = hashlib.sha256(combined.encode('utf-8')).digest()
        return int.from_bytes(hash_digest[:8], 'big')

    def generate_page(self, hex_id: str, page_num: int) -> str:
        seed = self._get_seed(hex_id, page_num)
        current = seed
        page = []

        for _ in range(self.page_length):
            current = (self.a * current + self.c) % self.modulus
            index = current % self.alphabet_size
            page.append(self.alphabet[index])

        return ''.join(page)
