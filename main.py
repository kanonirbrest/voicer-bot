import os
import time
import sys
import socket
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
import subprocess

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,  # Изменяем уровень на DEBUG
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Проверка на единственный экземпляр
def check_single_instance():
    try:
        logger.debug("Проверка на единственный экземпляр бота")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(('127.0.0.1', 5000))
        sock.close()
        logger.debug("Проверка пройдена успешно")
        return True
    except socket.error as e:
        logger.error(f"Ошибка при проверке экземпляра: {str(e)}")
        return False

# Загрузка переменных окружения
logger.debug("Загрузка переменных окружения")
load_dotenv()
TOKEN = os.getenv('TELEGRAM_TOKEN')
if not TOKEN:
    logger.error("Токен бота не найден в переменных окружения")
    sys.exit(1)
logger.debug("Токен бота загружен")

# Проверяем, не запущен ли уже бот
if not check_single_instance():
    logger.error("Бот уже запущен. Завершение работы.")
    sys.exit(1)

# Словарь с эффектами
EFFECTS = {
    'robot': 'Эффект робота',
    'rough': 'Грубый голос',
    'echo': 'Эффект эха',
    'slow': 'Замедление',
    'fast': 'Ускорение',
    'reverse': 'Обратный эффект'
}
logger.debug(f"Загружены эффекты: {EFFECTS}")

# Глобальный словарь для хранения голосовых сообщений
voice_messages = {}

# Время жизни сохраненного сообщения (в секундах)
MESSAGE_TIMEOUT = 300  # 5 минут

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    logger.info(f"Получена команда /start от пользователя {update.effective_user.id}")
    try:
        await update.message.reply_text(
            "Привет! Я бот для модификации голосовых сообщений.\n\n"
            "Как использовать:\n"
            "1. Ответьте на голосовое сообщение\n"
            "2. Выберите эффект\n"
            "3. Получите обработанное сообщение"
        )
        logger.info(f"Отправлено приветственное сообщение пользователю {update.effective_user.id}")
    except Exception as e:
        logger.error(f"Ошибка при обработке команды /start: {str(e)}")
        logger.exception("Полный стек ошибки:")

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик ответов на сообщения"""
    logger.info(f"Получен ответ от пользователя {update.effective_user.id}")
    logger.debug(f"Текст сообщения: {update.message.text}")
    logger.debug(f"Сообщение, на которое отвечают: {update.message.reply_to_message}")
    
    if not update.message.reply_to_message:
        logger.warning(f"Пользователь {update.effective_user.id} не ответил на сообщение")
        await update.message.reply_text("Пожалуйста, ответьте на сообщение.")
        return
    
    if not update.message.reply_to_message.voice:
        logger.warning(f"Пользователь {update.effective_user.id} ответил не на голосовое сообщение")
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    # Проверяем, содержит ли сообщение упоминание бота
    bot_username = context.bot.username
    if not bot_username:
        logger.error("Имя бота не найдено")
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")
        return
    
    bot_mention = f"@{bot_username}"
    logger.debug(f"Ожидаемое упоминание бота: {bot_mention}")
    
    if bot_mention.lower() not in update.message.text.lower():
        logger.warning(f"Пользователь {update.effective_user.id} не упомянул бота. Ожидалось: {bot_mention}")
        await update.message.reply_text(f"Пожалуйста, упомяните бота ({bot_mention}) в ответе на голосовое сообщение.")
        return
    
    logger.info(f"Пользователь {update.effective_user.id} правильно упомянул бота")
    
    try:
        # Сохраняем информацию о голосовом сообщении
        voice_info = {
            'file_id': update.message.reply_to_message.voice.file_id,
            'message_id': update.message.reply_to_message.message_id,
            'chat_id': update.message.chat_id,
            'reply_message_id': update.message.message_id,
            'timestamp': time.time()
        }
        logger.debug(f"Сохранена информация о голосовом сообщении: {voice_info}")
        
        # Сохраняем в глобальный словарь
        voice_messages[update.effective_user.id] = voice_info
        logger.debug(f"Текущее состояние voice_messages: {voice_messages}")
        
        # Создаем клавиатуру с эффектами
        keyboard = [
            [InlineKeyboardButton(effect_name, callback_data=effect_id)]
            for effect_id, effect_name in EFFECTS.items()
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        logger.debug("Создана клавиатура с эффектами")
        
        # Отправляем сообщение с кнопками
        await update.message.reply_text(
            "Выберите эффект для голосового сообщения:",
            reply_markup=reply_markup
        )
        logger.info(f"Отправлены кнопки эффектов пользователю {update.effective_user.id}")
    
    except Exception as e:
        logger.error(f"Ошибка в handle_reply: {str(e)}")
        logger.exception("Полный стек ошибки:")
        await update.message.reply_text("Произошла ошибка. Пожалуйста, попробуйте еще раз.")

async def apply_effect(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Применение выбранного эффекта"""
    query = update.callback_query
    await query.answer()
    
    effect = query.data
    user_id = update.effective_user.id
    logger.info(f"Пользователь {user_id} выбрал эффект: {effect}")
    
    # Получаем информацию о голосовом сообщении из глобального словаря
    if user_id not in voice_messages:
        logger.error(f"Информация о голосовом сообщении не найдена для пользователя {user_id}")
        logger.debug(f"Текущее состояние voice_messages: {voice_messages}")
        await query.message.edit_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    voice_info = voice_messages[user_id]
    logger.debug(f"Получена информация о голосовом сообщении: {voice_info}")
    
    # Проверяем, не истекло ли время действия сообщения
    if time.time() - voice_info['timestamp'] > MESSAGE_TIMEOUT:
        logger.warning(f"Время действия сообщения истекло для пользователя {user_id}")
        await query.message.edit_text("Время действия сообщения истекло. Пожалуйста, ответьте на голосовое сообщение снова.")
        del voice_messages[user_id]
        return
    
    logger.info(f"Обработка голосового сообщения с информацией: {voice_info}")
    
    try:
        # Получаем файл голосового сообщения
        try:
            voice = await context.bot.get_file(voice_info['file_id'])
            logger.debug(f"Получен файл голосового сообщения: {voice}")
        except Exception as e:
            logger.error(f"Ошибка при получении файла голосового сообщения: {str(e)}")
            await query.message.edit_text("Ошибка при получении голосового сообщения. Пожалуйста, попробуйте еще раз.")
            del voice_messages[user_id]
            return
        
        # Скачиваем файл
        with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_file:
            await voice.download_to_drive(temp_file.name)
            logger.debug(f"Файл скачан в: {temp_file.name}")
            
            # Конвертируем в WAV с помощью ffmpeg
            wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
            try:
                logger.debug("Начало конвертации в WAV с помощью ffmpeg")
                subprocess.run([
                    'ffmpeg', '-i', temp_file.name,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',
                    wav_file.name
                ], check=True)
                logger.info("Конвертация в WAV успешно завершена")
            except subprocess.CalledProcessError as e:
                logger.error(f"Ошибка при конвертации в WAV: {str(e)}")
                await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                return
            
            # Загружаем WAV файл
            wav_data, sample_rate = sf.read(wav_file.name)
            logger.debug(f"WAV файл загружен, частота дискретизации: {sample_rate}")
            
            # Применяем эффект
            logger.debug(f"Применение эффекта {effect}")
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
                logger.warning(f"Неизвестный эффект {effect} выбран пользователем {user_id}")
                await query.message.edit_text("Неизвестный эффект.")
                return
            
            logger.info(f"Эффект {effect} применен, новая частота дискретизации: {new_sample_rate}")
            
            # Сохраняем обработанный аудио
            with tempfile.NamedTemporaryFile(suffix='.ogg') as output_file:
                # Сначала сохраняем как WAV
                sf.write(output_file.name, processed_audio, new_sample_rate)
                logger.debug("Обработанный аудио сохранен как WAV")
                
                # Конвертируем в OGG с помощью ffmpeg
                ogg_file = tempfile.NamedTemporaryFile(suffix='.ogg')
                try:
                    logger.debug("Начало конвертации в OGG с помощью ffmpeg")
                    subprocess.run([
                        'ffmpeg', '-i', output_file.name,
                        '-acodec', 'libopus',
                        '-ar', '48000',
                        '-ac', '1',
                        ogg_file.name
                    ], check=True)
                    logger.info("Конвертация в OGG успешно завершена")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Ошибка при конвертации в OGG: {str(e)}")
                    await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                    return
                
                # Отправляем обработанное сообщение
                logger.debug(f"Отправка обработанного сообщения в чат {voice_info['chat_id']}")
                await context.bot.send_voice(
                    chat_id=voice_info['chat_id'],
                    voice=open(ogg_file.name, 'rb'),
                    caption=f"Эффект: {EFFECTS[effect]}",
                    reply_to_message_id=voice_info['message_id']
                )
                logger.info(f"Обработанное голосовое сообщение отправлено в чат {voice_info['chat_id']}")
                
                # Удаляем сообщение с кнопками
                await query.message.delete()
                logger.info("Сообщение с кнопками удалено")
                
                # Удаляем информацию о голосовом сообщении
                del voice_messages[user_id]
                logger.info(f"Информация о голосовом сообщении удалена для пользователя {user_id}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения для пользователя {user_id}: {str(e)}")
        logger.exception("Полный стек ошибки:")
        await query.message.edit_text("Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз.")
        # В случае ошибки также удаляем информацию о сообщении
        if user_id in voice_messages:
            del voice_messages[user_id]
            logger.info(f"Информация о голосовом сообщении удалена после ошибки для пользователя {user_id}")

def apply_robot_effect(audio, sample_rate):
    """Применение эффекта робота"""
    logger.debug("Применение эффекта робота")
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
    logger.debug("Применение эффекта грубого голоса")
    # Добавляем дисторшн
    processed_audio = np.tanh(audio * 5) * 0.5
    
    # Добавляем низкочастотную модуляцию
    t = np.arange(len(audio)) / sample_rate
    modulation = np.sin(2 * np.pi * 5 * t) * 0.3
    processed_audio = processed_audio * (1 + modulation)
    
    return processed_audio

def apply_echo_effect(audio, sample_rate):
    """Применение эффекта эха"""
    logger.debug("Применение эффекта эха")
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
    logger.debug("Применение эффекта замедления")
    # Уменьшаем частоту дискретизации в 2 раза
    new_sample_rate = sample_rate // 2
    # Изменяем длительность без изменения высоты тона
    processed_audio = signal.resample_poly(audio, 2, 1)
    return processed_audio, new_sample_rate

def apply_fast_effect(audio, sample_rate):
    """Применение эффекта ускорения"""
    logger.debug("Применение эффекта ускорения")
    # Увеличиваем частоту дискретизации в 2 раза
    new_sample_rate = sample_rate * 2
    # Изменяем длительность без изменения высоты тона
    processed_audio = signal.resample_poly(audio, 1, 2)
    return processed_audio, new_sample_rate

def apply_reverse_effect(audio, sample_rate):
    """Применение обратного эффекта"""
    logger.debug("Применение обратного эффекта")
    # Просто переворачиваем массив
    return np.flip(audio), sample_rate

def main():
    """Основная функция"""
    logger.info("Запуск бота")
    try:
        # Создаем приложение
        application = Application.builder().token(TOKEN).build()
        logger.debug("Приложение создано")
        
        # Добавляем обработчики
        application.add_handler(CommandHandler("start", start))
        application.add_handler(MessageHandler(filters.REPLY, handle_reply))
        application.add_handler(CallbackQueryHandler(apply_effect))
        logger.debug("Обработчики добавлены")
        
        # Запускаем бота
        logger.info("Бот запущен")
        application.run_polling()
    except Exception as e:
        logger.error(f"Ошибка при запуске бота: {str(e)}")
        logger.exception("Полный стек ошибки:")
        sys.exit(1)

if __name__ == '__main__':
    main() 