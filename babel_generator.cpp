#include <string>
#include <cstring>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>

extern "C" {

// Русский алфавит + знаки (в виде массива UTF-8 строк)
const char* alphabet[] = {
    " ", "а", "б", "в", "г", "д", "е", "ё", "ж", "з",
    "и", "й", "к", "л", "м", "н", "о", "п", "р", "с",
    "т", "у", "ф", "х", "ц", "ч", "ш", "щ", "ъ", "ы",
    "ь", "э", "ю", "я", ".", ","
};

const int alphabet_size = sizeof(alphabet) / sizeof(alphabet[0]);

uint64_t hex_to_seed(const std::string& hex) {
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256((unsigned char*)hex.c_str(), hex.size(), hash);
    uint64_t seed = 0;
    for (int i = 0; i < 8; ++i) {
        seed = (seed << 8) | hash[i];
    }
    return seed;
}

char* generate_page(const char* hex_id_c, int page_num) {
    std::string hex_id(hex_id_c);
    uint64_t seed = hex_to_seed(hex_id + std::to_string(page_num));

    const uint64_t a = 1664525;
    const uint64_t c = 1013904223;
    const uint64_t mod = 4294967296;

    uint64_t state = seed;
    std::string output;

    int char_count = 0;
    while (char_count < 3200) {
        state = (a * state + c) % mod;
        int idx = state % alphabet_size;
        output += alphabet[idx];
        ++char_count;
    }

    char* result = new char[output.size() + 1];
    std::memcpy(result, output.c_str(), output.size());
    result[output.size()] = '\0';
    return result;
}

}
