# Voicer Bot

Telegram бот для модификации голосовых сообщений.

## Функциональность

- Обработка голосовых сообщений
- Применение различных эффектов:
  - Профессиональный автотюн
  - Музыкальный автотюн
  - Эффект робота
  - Грубый голос

## Как использовать

1. Найдите голосовое сообщение в любом чате
2. Ответьте на него (reply)
3. Бот предложит выбрать эффект
4. Выберите нужный эффект
5. Получите обработанное сообщение

## Установка

1. Клонируйте репозиторий
2. Установите зависимости:
```bash
pip install -r requirements.txt
```
3. Создайте файл `.env` и добавьте токен бота:
```
TELEGRAM_TOKEN=ваш_токен_бота
```
4. Запустите бота:
```bash
python main.py
```

## Развертывание на Replit

1. Импортируйте репозиторий в Replit
2. Добавьте токен бота в Secrets
3. Запустите бота 