import os
import tempfile
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters
from pydub import AudioSegment
import numpy as np
import soundfile as sf
import librosa
import pyworld as pw
import crepe
from scipy.signal import resample_poly

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Эффекты для голоса
EFFECTS = {
    'autotune': 'Профессиональный автотюн',
    'robot': 'Робот',
    'deep': 'Грубый голос',
    'musical': 'Музыкальный автотюн'
}

# Музыкальные ноты и их частоты
NOTES = {
    'C': 261.63,
    'C#': 277.18,
    'D': 293.66,
    'D#': 311.13,
    'E': 329.63,
    'F': 349.23,
    'F#': 369.99,
    'G': 392.00,
    'G#': 415.30,
    'A': 440.00,
    'A#': 466.16,
    'B': 493.88
}

def find_closest_note(frequency):
    """Находит ближайшую ноту к заданной частоте"""
    closest_note = None
    min_diff = float('inf')
    
    for note, note_freq in NOTES.items():
        diff = abs(frequency - note_freq)
        if diff < min_diff:
            min_diff = diff
            closest_note = note_freq
    
    return closest_note

def apply_professional_autotune(audio_path):
    """Применение профессионального автотюна с использованием pyworld"""
    # Загружаем аудио
    y, sr = librosa.load(audio_path)
    
    # Определяем высоту тона с помощью crepe
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
    
    # Используем WORLD для анализа и синтеза
    _f0, t = pw.dio(y.astype(np.float64), sr)  # Определение F0
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)  # Уточнение F0
    sp = pw.cheaptrick(y.astype(np.float64), f0, t, sr)  # Спектральный анализ
    ap = pw.d4c(y.astype(np.float64), f0, t, sr)  # Апериодический анализ
    
    # Корректируем F0 до ближайшей ноты
    for i in range(len(f0)):
        if f0[i] > 0:  # Игнорируем тишину
            closest_note = find_closest_note(f0[i])
            f0[i] = closest_note
    
    # Синтезируем новый звук
    y_modified = pw.synthesize(f0, sp, ap, sr)
    
    # Нормализуем
    y_modified = librosa.util.normalize(y_modified)
    
    # Сохраняем результат
    sf.write(audio_path, y_modified, sr)
    return audio_path

def apply_musical_autotune(audio_path):
    """Применение музыкального автотюна с повышением голоса"""
    # Загружаем аудио
    y, sr = librosa.load(audio_path)
    
    # Определяем высоту тона с помощью crepe
    time, frequency, confidence, activation = crepe.predict(y, sr, viterbi=True)
    
    # Используем WORLD для анализа и синтеза
    _f0, t = pw.dio(y.astype(np.float64), sr)
    f0 = pw.stonemask(y.astype(np.float64), _f0, t, sr)
    sp = pw.cheaptrick(y.astype(np.float64), f0, t, sr)
    ap = pw.d4c(y.astype(np.float64), f0, t, sr)
    
    # Корректируем F0 до ближайшей ноты и повышаем на октаву
    for i in range(len(f0)):
        if f0[i] > 0:
            closest_note = find_closest_note(f0[i])
            f0[i] = closest_note * 2  # Повышаем на октаву
    
    # Добавляем вибрато
    t = np.arange(len(f0)) / sr
    vibrato = 0.1 * np.sin(2 * np.pi * 5 * t)
    f0 = f0 * (1 + vibrato)
    
    # Синтезируем новый звук
    y_modified = pw.synthesize(f0, sp, ap, sr)
    
    # Добавляем эхо
    y_modified = librosa.effects.preemphasis(y_modified)
    
    # Нормализуем
    y_modified = librosa.util.normalize(y_modified)
    
    # Сохраняем результат
    sf.write(audio_path, y_modified, sr)
    return audio_path

def apply_robot_effect(audio_path):
    """Применение эффекта робота"""
    data, samplerate = sf.read(audio_path)
    # Добавляем эффект робота через модуляцию
    t = np.arange(len(data)) / samplerate
    modulation = np.sin(2 * np.pi * 5 * t)
    modified_data = data * (1 + 0.3 * modulation)
    sf.write(audio_path, modified_data, samplerate)
    return audio_path

def apply_deep_voice(audio_path):
    """Применение эффекта грубого голоса"""
    data, samplerate = sf.read(audio_path)
    # Понижаем частоту для эффекта грубого голоса
    modified_data = np.interp(
        np.arange(0, len(data), 0.7),
        np.arange(len(data)),
        data
    )
    sf.write(audio_path, modified_data, samplerate)
    return audio_path

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    await update.message.reply_text(
        "Привет! Я бот для модификации голосовых сообщений. "
        "Ответьте на голосовое сообщение, и я предложу варианты его модификации."
    )

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответов на сообщения"""
    # Проверяем, что это ответ на сообщение
    if not update.message.reply_to_message:
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Проверяем, что это ответ на голосовое сообщение
    if not update.message.reply_to_message.voice:
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Получаем голосовое сообщение
    voice = update.message.reply_to_message.voice
    
    # Сохраняем информацию о сообщении в контексте
    context.user_data['voice_message'] = voice
    context.user_data['reply_to_message_id'] = update.message.reply_to_message.message_id
    
    # Создаем клавиатуру с эффектами
    keyboard = [
        [InlineKeyboardButton(EFFECTS['autotune'], callback_data='autotune')],
        [InlineKeyboardButton(EFFECTS['musical'], callback_data='musical')],
        [InlineKeyboardButton(EFFECTS['robot'], callback_data='robot')],
        [InlineKeyboardButton(EFFECTS['deep'], callback_data='deep')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Выберите эффект для голосового сообщения:",
        reply_markup=reply_markup
    )

async def apply_effect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Применение выбранного эффекта к голосовому сообщению"""
    query = update.callback_query
    await query.answer()
    
    effect = query.data
    voice = context.user_data.get('voice_message')
    reply_to_message_id = context.user_data.get('reply_to_message_id')
    
    if not voice or not reply_to_message_id:
        await query.edit_message_text("Извините, произошла ошибка. Попробуйте отправить сообщение снова.")
        return
    
    # Скачиваем голосовое сообщение
    voice_file = await context.bot.get_file(voice.file_id)
    
    # Создаем временный файл для аудио
    with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_file:
        await voice_file.download_to_drive(temp_file.name)
        
        # Конвертируем в WAV для обработки
        audio = AudioSegment.from_ogg(temp_file.name)
        audio.export("temp.wav", format="wav")
        
        # Применяем эффект
        if effect == 'autotune':
            modified_audio = apply_professional_autotune("temp.wav")
        elif effect == 'musical':
            modified_audio = apply_musical_autotune("temp.wav")
        elif effect == 'robot':
            modified_audio = apply_robot_effect("temp.wav")
        elif effect == 'deep':
            modified_audio = apply_deep_voice("temp.wav")
        
        # Отправляем модифицированное сообщение
        with open("temp.wav", "rb") as audio_file:
            await context.bot.send_voice(
                chat_id=update.effective_chat.id,
                voice=audio_file,
                reply_to_message_id=reply_to_message_id
            )
    
    # Удаляем временные файлы
    os.remove("temp.wav")
    await query.edit_message_text(f"Эффект {EFFECTS[effect]} применен!")

def main():
    """Запуск бота"""
    application = Application.builder().token(TOKEN).build()

    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.REPLY & filters.TEXT, handle_reply))
    application.add_handler(CallbackQueryHandler(apply_effect))

    # Запускаем бота
    application.run_polling()

if __name__ == '__main__':
    main() 