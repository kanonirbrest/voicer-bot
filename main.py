import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, InlineQueryHandler
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import io
import tempfile
from scipy import signal

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Словарь с эффектами
EFFECTS = {
    'robot': 'Эффект робота',
    'rough': 'Грубый голос',
    'echo': 'Эффект эха'
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Я бот для модификации голосовых сообщений.\n\n"
        "Как использовать:\n"
        "1. Ответьте на голосовое сообщение\n"
        "2. Выберите эффект\n"
        "3. Получите обработанное сообщение\n\n"
        "Или используйте меня в инлайн-режиме: @voicer_132_bot"
    )

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик инлайн-запросов"""
    query = update.inline_query.query.lower()
    results = []
    
    # Если запрос пустой, показываем все эффекты
    if not query:
        for effect_id, effect_name in EFFECTS.items():
            results.append(
                InlineQueryResultArticle(
                    id=effect_id,
                    title=effect_name,
                    description=f'Применить эффект {effect_name}',
                    input_message_content=InputTextMessageContent(
                        message_text=f"Выбран эффект: {effect_name}\n\nОтправьте голосовое сообщение для обработки."
                    )
                )
            )
    else:
        # Ищем эффекты по запросу
        for effect_id, effect_name in EFFECTS.items():
            if query in effect_name.lower() or query in effect_id:
                results.append(
                    InlineQueryResultArticle(
                        id=effect_id,
                        title=effect_name,
                        description=f'Применить эффект {effect_name}',
                        input_message_content=InputTextMessageContent(
                            message_text=f"Выбран эффект: {effect_name}\n\nОтправьте голосовое сообщение для обработки."
                        )
                    )
                )
    
    await update.inline_query.answer(results)

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответов на сообщения"""
    if not update.message.reply_to_message or not update.message.reply_to_message.voice:
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Проверяем, содержит ли сообщение упоминание бота
    if '@voicer_132_bot' not in update.message.text.lower():
        await update.message.reply_text("Пожалуйста, упомяните бота (@voicer_132_bot) в ответе на голосовое сообщение.")
        return
    
    # Сохраняем информацию о голосовом сообщении в контекст
    context.user_data['voice_message'] = {
        'file_id': update.message.reply_to_message.voice.file_id,
        'message_id': update.message.reply_to_message.message_id
    }
    
    # Создаем клавиатуру с эффектами
    keyboard = [
        [InlineKeyboardButton(effect_name, callback_data=effect_id)]
        for effect_id, effect_name in EFFECTS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    # Редактируем текущее сообщение, добавляя кнопки
    await update.message.edit_text(
        "Выберите эффект для голосового сообщения:",
        reply_markup=reply_markup
    )

async def apply_effect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Применение выбранного эффекта"""
    query = update.callback_query
    await query.answer()
    
    effect = query.data
    message = query.message
    
    # Получаем голосовое сообщение из реплая
    if not message.reply_to_message or not message.reply_to_message.voice:
        await message.edit_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    voice = message.reply_to_message.voice
    file = await context.bot.get_file(voice.file_id)
    
    # Скачиваем файл
    with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_file:
        await file.download_to_drive(temp_file.name)
        
        # Конвертируем в WAV
        audio = AudioSegment.from_ogg(temp_file.name)
        wav_data = np.array(audio.get_array_of_samples())
        sample_rate = audio.frame_rate
        
        # Применяем эффект
        if effect == 'robot':
            processed_audio = apply_robot_effect(wav_data, sample_rate)
        elif effect == 'rough':
            processed_audio = apply_rough_voice(wav_data, sample_rate)
        elif effect == 'echo':
            processed_audio = apply_echo_effect(wav_data, sample_rate)
        else:
            await message.edit_text("Неизвестный эффект.")
            return
        
        # Сохраняем обработанный аудио
        with tempfile.NamedTemporaryFile(suffix='.ogg') as output_file:
            sf.write(output_file.name, processed_audio, sample_rate)
            audio = AudioSegment.from_wav(output_file.name)
            audio.export(output_file.name, format="ogg")
            
            # Отправляем обработанное сообщение
            await message.reply_to_message.reply_voice(
                voice=open(output_file.name, 'rb'),
                caption=f"Эффект: {EFFECTS[effect]}"
            )
            
            # Удаляем сообщение с кнопками
            await message.delete()

def apply_robot_effect(audio, sample_rate):
    """Применение эффекта робота"""
    # Добавляем фазовое искажение
    processed_audio = np.zeros_like(audio)
    for i in range(len(audio)):
        processed_audio[i] = np.sin(audio[i] * 10) * 0.5
    
    # Добавляем шум
    noise = np.random.normal(0, 0.1, len(audio))
    processed_audio = processed_audio + noise
    
    return processed_audio

def apply_rough_voice(audio, sample_rate):
    """Применение эффекта грубого голоса"""
    # Добавляем дисторшн
    processed_audio = np.tanh(audio * 5) * 0.5
    
    # Добавляем низкочастотную модуляцию
    t = np.arange(len(audio)) / sample_rate
    modulation = np.sin(2 * np.pi * 5 * t) * 0.3
    processed_audio = processed_audio * (1 + modulation)
    
    return processed_audio

def apply_echo_effect(audio, sample_rate):
    """Применение эффекта эха"""
    # Задержка в секундах
    delay = 0.3
    # Количество повторов
    repeats = 3
    # Затухание
    decay = 0.5
    
    # Создаем задержанные копии
    processed_audio = np.copy(audio)
    delay_samples = int(delay * sample_rate)
    
    for i in range(1, repeats + 1):
        # Создаем задержанную копию
        delayed = np.zeros_like(audio)
        delayed[i * delay_samples:] = audio[:-i * delay_samples] if i * delay_samples < len(audio) else 0
        # Добавляем с затуханием
        processed_audio += delayed * (decay ** i)
    
    # Нормализуем
    processed_audio = processed_audio / np.max(np.abs(processed_audio))
    
    return processed_audio

def main():
    """Основная функция"""
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.REPLY, handle_reply))
    application.add_handler(CallbackQueryHandler(apply_effect))
    application.add_handler(InlineQueryHandler(inline_query))
    
    # Запускаем бота
    print("Бот запущен")
    application.run_polling()

if __name__ == '__main__':
    main() 