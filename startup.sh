#!/bin/bash
set -e

echo "🚀 Запуск Babel Semantic Scanner"
echo "⏳ Обновляем систему..."
apt-get update -y && apt-get install -y git python3-pip

echo "📦 Клонируем репозиторий..."
if [ ! -d "babel_semantic_scanner" ]; then
    git clone https://github.com/mkiber/babel_semantic_scanner.git
fi
cd babel_semantic_scanner

echo "🐍 Устанавливаем зависимости..."
pip install -r requirements.txt

echo "🧠 Запускаем бесконечный смысловой сканер..."
nohup python3 infinite_scanner.py > scanner_output.log 2>&1 &

echo "✅ Всё запущено! Логи доступны через:"
echo "   tail -f scanner_output.log"
echo "   tail -f scanner.log"
tail -f scanner_output.log
