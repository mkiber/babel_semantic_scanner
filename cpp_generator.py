import ctypes
import os

lib_path = os.path.abspath("./libbabel.so")
lib = ctypes.CDLL(lib_path)

lib.generate_page.argtypes = [ctypes.c_char_p, ctypes.c_int]
lib.generate_page.restype = ctypes.c_char_p


def cpp_generate_page(hex_id: str, page_num: int) -> str:
    # Вместо "string_at(ptr)", считываем ровно 3200 символов × max 2 байта = 6400
    # (UTF-8 в кириллице обычно 2 байта на символ)
    ptr = lib.generate_page(hex_id.encode("utf-8"), page_num) # 6400 — с запасом
    raw_bytes = ctypes.string_at(ptr, 6400)
    safe_bytes = raw_bytes.split(b'\0')[0]
    return safe_bytes.decode("utf-8", errors="replace")
