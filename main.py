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
import logging

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Загрузка переменных окружения
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')

# Словарь с эффектами
EFFECTS = {
    'robot': 'Эффект робота',
    'rough': 'Грубый голос',
    'echo': 'Эффект эха',
    'slow': 'Замедление',
    'fast': 'Ускорение',
    'reverse': 'Обратный эффект'
}

# Глобальный словарь для хранения голосовых сообщений
voice_messages = {}

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    logger.info(f"User {update.effective_user.id} started the bot")
    await update.message.reply_text(
        "Привет! Я бот для модификации голосовых сообщений.\n\n"
        "Как использовать:\n"
        "1. Ответьте на голосовое сообщение\n"
        "2. Выберите эффект\n"
        "3. Получите обработанное сообщение"
    )

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответов на сообщения"""
    logger.info(f"Received reply from user {update.effective_user.id}")
    logger.info(f"Message text: {update.message.text}")
    logger.info(f"Reply to message: {update.message.reply_to_message}")
    
    if not update.message.reply_to_message or not update.message.reply_to_message.voice:
        logger.warning(f"User {update.effective_user.id} replied to non-voice message")
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Проверяем, содержит ли сообщение упоминание бота
    bot_username = context.bot.username
    if not bot_username:
        logger.error("Bot username not found")
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
        return
    
    bot_mention = f"@{bot_username}"
    if bot_mention.lower() not in update.message.text.lower():
        logger.warning(f"User {update.effective_user.id} didn't mention the bot. Expected: {bot_mention}")
        await update.message.reply_text(f"Пожалуйста, упомяните бота ({bot_mention}) в ответе на голосовое сообщение.")
        return
    
    logger.info(f"User {update.effective_user.id} mentioned the bot correctly")
    
    try:
        # Сохраняем информацию о голосовом сообщении
        voice_info = {
            'file_id': update.message.reply_to_message.voice.file_id,
            'message_id': update.message.reply_to_message.message_id,
            'chat_id': update.message.chat_id,
            'reply_message_id': update.message.message_id  # Сохраняем ID сообщения с упоминанием бота
        }
        logger.info(f"Saving voice message info: {voice_info}")
        
        # Сохраняем в глобальный словарь
        voice_messages[update.effective_user.id] = voice_info
        
        # Создаем клавиатуру с эффектами
        keyboard = [
            [InlineKeyboardButton(effect_name, callback_data=effect_id)]
            for effect_id, effect_name in EFFECTS.items()
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Отправляем сообщение с кнопками
        await update.message.reply_text(
            "Выберите эффект для голосового сообщения:",
            reply_markup=reply_markup
        )
        logger.info(f"Sent effect buttons to user {update.effective_user.id}")
    
    except Exception as e:
        logger.error(f"Error in handle_reply: {str(e)}")
        logger.exception("Full traceback:")
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")

async def apply_effect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Применение выбранного эффекта"""
    query = update.callback_query
    await query.answer()
    
    effect = query.data
    user_id = update.effective_user.id
    logger.info(f"User {user_id} selected effect: {effect}")
    
    # Получаем информацию о голосовом сообщении из глобального словаря
    if user_id not in voice_messages:
        logger.error(f"Voice message info not found for user {user_id}")
        await query.message.edit_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    voice_info = voice_messages[user_id]
    logger.info(f"Processing voice message with info: {voice_info}")
    
    try:
        voice = await context.bot.get_file(voice_info['file_id'])
        logger.info(f"Got voice file: {voice}")
        
        # Скачиваем файл
        with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_file:
            await voice.download_to_drive(temp_file.name)
            logger.info(f"Downloaded voice file to {temp_file.name}")
            
            # Конвертируем в WAV
            audio = AudioSegment.from_ogg(temp_file.name)
            wav_data = np.array(audio.get_array_of_samples())
            sample_rate = audio.frame_rate
            logger.info(f"Converted audio to WAV, sample rate: {sample_rate}")
            
            # Применяем эффект
            if effect == 'robot':
                processed_audio = apply_robot_effect(wav_data, sample_rate)
                new_sample_rate = sample_rate
            elif effect == 'rough':
                processed_audio = apply_rough_voice(wav_data, sample_rate)
                new_sample_rate = sample_rate
            elif effect == 'echo':
                processed_audio = apply_echo_effect(wav_data, sample_rate)
                new_sample_rate = sample_rate
            elif effect == 'slow':
                processed_audio, new_sample_rate = apply_slow_effect(wav_data, sample_rate)
            elif effect == 'fast':
                processed_audio, new_sample_rate = apply_fast_effect(wav_data, sample_rate)
            elif effect == 'reverse':
                processed_audio, new_sample_rate = apply_reverse_effect(wav_data, sample_rate)
            else:
                logger.warning(f"Unknown effect {effect} selected by user {user_id}")
                await query.message.edit_text("Неизвестный эффект.")
                return
            
            logger.info(f"Applied effect {effect}, new sample rate: {new_sample_rate}")
            
            # Сохраняем обработанный аудио
            with tempfile.NamedTemporaryFile(suffix='.ogg') as output_file:
                sf.write(output_file.name, processed_audio, new_sample_rate)
                audio = AudioSegment.from_wav(output_file.name)
                audio.export(output_file.name, format="ogg")
                logger.info(f"Saved processed audio to {output_file.name}")
                
                # Отправляем обработанное сообщение
                await context.bot.send_voice(
                    chat_id=voice_info['chat_id'],
                    voice=open(output_file.name, 'rb'),
                    caption=f"Эффект: {EFFECTS[effect]}",
                    reply_to_message_id=voice_info['message_id']
                )
                logger.info(f"Sent processed voice message to chat {voice_info['chat_id']}")
                
                # Удаляем сообщение с кнопками
                await query.message.delete()
                logger.info(f"Deleted buttons message")
                
                # Удаляем информацию о голосовом сообщении
                del voice_messages[user_id]
                logger.info(f"Removed voice message info for user {user_id}")
    
    except Exception as e:
        logger.error(f"Error processing voice message for user {user_id}: {str(e)}")
        logger.exception("Full traceback:")
        await query.message.edit_text("Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз.")

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

def apply_slow_effect(audio, sample_rate):
    """Применение эффекта замедления"""
    # Уменьшаем частоту дискретизации в 2 раза
    new_sample_rate = sample_rate // 2
    # Изменяем длительность без изменения высоты тона
    processed_audio = signal.resample_poly(audio, 2, 1)
    return processed_audio, new_sample_rate

def apply_fast_effect(audio, sample_rate):
    """Применение эффекта ускорения"""
    # Увеличиваем частоту дискретизации в 2 раза
    new_sample_rate = sample_rate * 2
    # Изменяем длительность без изменения высоты тона
    processed_audio = signal.resample_poly(audio, 1, 2)
    return processed_audio, new_sample_rate

def apply_reverse_effect(audio, sample_rate):
    """Применение обратного эффекта"""
    # Просто переворачиваем массив
    return np.flip(audio), sample_rate

def main():
    """Основная функция"""
    # Создаем приложение
    application = Application.builder().token(TOKEN).build()
    
    # Добавляем обработчики
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.REPLY, handle_reply))
    application.add_handler(CallbackQueryHandler(apply_effect))
    
    # Запускаем бота
    print("Бот запущен")
    application.run_polling()

if __name__ == '__main__':
    main() 