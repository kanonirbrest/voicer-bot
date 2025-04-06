#!/bin/bash

# Установка системных зависимостей
apt-get update
apt-get install -y ffmpeg portaudio19-dev libsndfile1

# Создание виртуального окружения
python -m venv venv
source venv/bin/activate

# Установка Python-пакетов
pip install -r requirements.txt 