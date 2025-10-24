#!/bin/bash

# Явный путь к openssl (M1/M2 macOS по умолчанию)
OPENSSL_PREFIX="/opt/homebrew/opt/openssl@3"

# Проверка наличия OpenSSL
if [ ! -d "$OPENSSL_PREFIX" ]; then
  echo "❌ OpenSSL не найден в $OPENSSL_PREFIX. Установи его через: brew install openssl"
  exit 1
fi

echo "🛠️ Компиляция babel_generator.cpp → libbabel.so"
g++ -fPIC -shared babel_generator.cpp -o libbabel.so \
-I$OPENSSL_PREFIX/include \
-L$OPENSSL_PREFIX/lib \
-lssl -lcrypto

if [ $? -eq 0 ]; then
  echo "✅ Успешно собрано: libbabel.so"
else
  echo "❌ Ошибка при компиляции"
  exit 1
fi
