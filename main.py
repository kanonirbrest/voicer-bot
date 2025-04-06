import os
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import librosa
import io
import tempfile
from scipy import signal
import pyworld as pw
import crepe
from scipy.signal import resample_poly

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Эффекты для голоса
EFFECTS = {
    'autotune': 'Профессиональный автотюн',
    'musical': 'Музыкальный автотюн',
    'robot': 'Эффект робота',
    'rough': 'Грубый голос'
}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Я бот для модификации голосовых сообщений.\n\n"
        "Как использовать:\n"
        "1. Ответьте на голосовое сообщение\n"
        "2. Выберите эффект\n"
        "3. Получите обработанное сообщение\n\n"
        "Или используйте меня в инлайн-режиме: @voicer_132_bot эффект"
    )

async def inline_query(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик инлайн-запросов"""
    query = update.inline_query.query.lower()
    results = []
    
    # Если запрос пустой, показываем все эффекты
    if not query:
        for effect_id, effect_name in EFFECTS.items():
            results.append({
                'type': 'article',
                'id': effect_id,
                'title': effect_name,
                'description': f'Применить эффект {effect_name}',
                'input_message_content': {
                    'message_text': f'Выбран эффект: {effect_name}\n\nОтправьте голосовое сообщение для обработки.'
                }
            })
    else:
        # Ищем эффекты по запросу
        for effect_id, effect_name in EFFECTS.items():
            if query in effect_name.lower() or query in effect_id:
                results.append({
                    'type': 'article',
                    'id': effect_id,
                    'title': effect_name,
                    'description': f'Применить эффект {effect_name}',
                    'input_message_content': {
                        'message_text': f'Выбран эффект: {effect_name}\n\nОтправьте голосовое сообщение для обработки.'
                    }
                })
    
    await update.inline_query.answer(results)

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответов на сообщения"""
    if not update.message.reply_to_message or not update.message.reply_to_message.voice:
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Создаем клавиатуру с эффектами
    keyboard = [
        [InlineKeyboardButton(effect_name, callback_data=effect_id)]
        for effect_id, effect_name in EFFECTS.items()
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
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
        if effect == 'autotune':
            processed_audio = apply_autotune(wav_data, sample_rate)
        elif effect == 'musical':
            processed_audio = apply_musical_autotune(wav_data, sample_rate)
        elif effect == 'robot':
            processed_audio = apply_robot_effect(wav_data, sample_rate)
        elif effect == 'rough':
            processed_audio = apply_rough_voice(wav_data, sample_rate)
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

def apply_autotune(audio, sample_rate):
    """Применение профессионального автотюна"""
    # Используем CREPE для определения высоты тона
    time, frequency, confidence, activation = crepe.predict(audio, sample_rate, viterbi=True)
    
    # Создаем синусоидальный сигнал с нужной частотой
    t = np.arange(len(audio)) / sample_rate
    processed_audio = np.zeros_like(audio, dtype=np.float32)
    
    for i in range(len(time)):
        if confidence[i] > 0.5:  # Используем только уверенные предсказания
            freq = frequency[i]
            if freq > 0:  # Игнорируем немые участки
                start_idx = int(time[i] * sample_rate)
                end_idx = min(int(time[i+1] * sample_rate) if i+1 < len(time) else len(audio), len(audio))
                duration = (end_idx - start_idx) / sample_rate
                
                if duration > 0:
                    t_segment = np.arange(start_idx, end_idx) / sample_rate
                    segment = np.sin(2 * np.pi * freq * t_segment)
                    processed_audio[start_idx:end_idx] = segment * 0.5
    
    return processed_audio

def apply_musical_autotune(audio, sample_rate):
    """Применение музыкального автотюна"""
    # Определяем ноты
    pitches, magnitudes = librosa.piptrack(y=audio, sr=sample_rate)
    
    # Находим доминирующую частоту для каждого кадра
    pitch = []
    for t in range(pitches.shape[1]):
        index = magnitudes[:, t].argmax()
        pitch.append(pitches[index, t])
    
    # Квантуем частоты к ближайшей ноте
    notes = librosa.hz_to_note(np.array(pitch))
    quantized_freqs = librosa.note_to_hz(notes)
    
    # Создаем новый сигнал
    t = np.arange(len(audio)) / sample_rate
    processed_audio = np.zeros_like(audio)
    
    frame_length = len(audio) // len(quantized_freqs)
    for i, freq in enumerate(quantized_freqs):
        if not np.isnan(freq):
            start = i * frame_length
            end = min((i + 1) * frame_length, len(audio))
            t_segment = t[start:end]
            processed_audio[start:end] = np.sin(2 * np.pi * freq * t_segment) * 0.5
    
    return processed_audio

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