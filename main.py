import os
import time
import sys
import socket
from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InlineQueryResultArticle, InputTextMessageContent
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes, InlineQueryHandler
from pydub import AudioSegment
from pydub.effects import speedup, pitch_shift, normalize, compress_dynamic_range
from pydub.generators import Sine, Square, WhiteNoise
import numpy as np
import soundfile as sf
import io
import tempfile
from scipy import signal
import logging
import subprocess
import librosa
import pyworld as pw
from scipy.interpolate import interp1d
import scipy

# Настройка логирования
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG,  # Изменяем уровень на DEBUG
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)

# Добавляем логирование для HTTP-запросов
logging.getLogger('httpx').setLevel(logging.DEBUG)
logging.getLogger('httpcore').setLevel(logging.DEBUG)
logging.getLogger('telegram').setLevel(logging.DEBUG)
logging.getLogger('telegram.ext').setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Проверка наличия ffmpeg
def check_ffmpeg():
    try:
        logger.debug("Проверка наличия ffmpeg")
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            logger.info("ffmpeg найден и работает")
            return True
        else:
            logger.error(f"ffmpeg не работает: {result.stderr}")
            return False
    except FileNotFoundError:
        logger.error("ffmpeg не установлен")
        return False

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

# Проверяем наличие ffmpeg
if not check_ffmpeg():
    logger.error("ffmpeg не установлен или не работает. Установите ffmpeg и перезапустите бота.")
    sys.exit(1)

# Проверяем, не запущен ли уже бот
if not check_single_instance():
    logger.error("Бот уже запущен. Завершение работы.")
    sys.exit(1)

# Словарь с эффектами
EFFECTS = {
    'robot': 'Эффект робота',
    'musical': 'Музыкальный эффект голоса',
    'autotune': 'Эффект автотюна',
    'rough': 'Эффект грубого голоса'
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
    logger.debug(f"Полная информация о callback_query: {query.to_dict()}")
    logger.debug(f"Текущее состояние контекста: {context.user_data}")
    
    # Получаем информацию о голосовом сообщении из глобального словаря
    if user_id not in voice_messages:
        logger.error(f"Информация о голосовом сообщении не найдена для пользователя {user_id}")
        logger.debug(f"Текущее состояние voice_messages: {voice_messages}")
        logger.debug(f"Все ключи в voice_messages: {list(voice_messages.keys())}")
        logger.debug(f"Текущий user_id: {user_id}, тип: {type(user_id)}")
        await query.message.edit_text("Пожалуйста, ответьте на голосовое сообщение.")
        return
    
    voice_info = voice_messages[user_id]
    logger.debug(f"Получена информация о голосовом сообщении: {voice_info}")
    logger.debug(f"Типы данных в voice_info: { {k: type(v) for k, v in voice_info.items()} }")
    logger.debug(f"Полная информация о сообщении: {query.message.to_dict()}")
    
    # Проверяем, не истекло ли время действия сообщения
    current_time = time.time()
    message_age = current_time - voice_info['timestamp']
    logger.debug(f"Текущее время: {current_time}, время сообщения: {voice_info['timestamp']}, возраст: {message_age} сек")
    logger.debug(f"Таймаут сообщения: {MESSAGE_TIMEOUT} сек")
    
    if message_age > MESSAGE_TIMEOUT:
        logger.warning(f"Время действия сообщения истекло для пользователя {user_id}")
        logger.debug(f"Возраст сообщения: {message_age} сек, таймаут: {MESSAGE_TIMEOUT} сек")
        await query.message.edit_text("Время действия сообщения истекло. Пожалуйста, ответьте на голосовое сообщение снова.")
        del voice_messages[user_id]
        return
    
    logger.info(f"Обработка голосового сообщения с информацией: {voice_info}")
    
    try:
        # Получаем файл голосового сообщения
        try:
            logger.debug(f"Попытка получить файл с file_id: {voice_info['file_id']}")
            logger.debug(f"Тип file_id: {type(voice_info['file_id'])}")
            voice = await context.bot.get_file(voice_info['file_id'])
            logger.debug(f"Получен файл голосового сообщения: {voice}")
            logger.debug(f"Информация о файле: {voice.to_dict()}")
            logger.debug(f"Размер файла: {voice.file_size} байт")
        except Exception as e:
            logger.error(f"Ошибка при получении файла голосового сообщения: {str(e)}")
            logger.exception("Полный стек ошибки при получении файла:")
            logger.debug(f"Текущее состояние voice_messages: {voice_messages}")
            await query.message.edit_text("Ошибка при получении голосового сообщения. Пожалуйста, попробуйте еще раз.")
            del voice_messages[user_id]
            return
        
        # Скачиваем файл
        with tempfile.NamedTemporaryFile(suffix='.ogg') as temp_file:
            try:
                logger.debug(f"Начало скачивания файла в: {temp_file.name}")
                logger.debug(f"Права доступа к временной директории: {os.stat(os.path.dirname(temp_file.name)).st_mode}")
                await voice.download_to_drive(temp_file.name)
                logger.debug(f"Файл скачан в: {temp_file.name}")
                
                # Проверяем размер файла
                file_size = os.path.getsize(temp_file.name)
                logger.debug(f"Размер файла: {file_size} байт")
                if file_size == 0:
                    logger.error("Скачанный файл пуст")
                    raise Exception("Скачанный файл пуст")
                
                # Проверяем права доступа к файлу
                file_permissions = os.stat(temp_file.name).st_mode
                logger.debug(f"Права доступа к файлу: {file_permissions}")
                logger.debug(f"Владелец файла: {os.stat(temp_file.name).st_uid}")
                logger.debug(f"Группа файла: {os.stat(temp_file.name).st_gid}")
                
            except Exception as e:
                logger.error(f"Ошибка при скачивании файла: {str(e)}")
                logger.exception("Полный стек ошибки при скачивании:")
                logger.debug(f"Текущее состояние временного файла: {os.path.exists(temp_file.name)}")
                await query.message.edit_text("Ошибка при скачивании голосового сообщения. Пожалуйста, попробуйте еще раз.")
                return
            
            # Конвертируем в WAV с помощью ffmpeg
            wav_file = tempfile.NamedTemporaryFile(suffix='.wav')
            try:
                logger.debug("Начало конвертации в WAV с помощью ffmpeg")
                logger.debug(f"Исходный файл: {temp_file.name}")
                logger.debug(f"Целевой файл: {wav_file.name}")
                
                result = subprocess.run([
                    'ffmpeg', '-y', '-i', temp_file.name,
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',
                    wav_file.name
                ], capture_output=True, text=True)
                
                logger.debug(f"Вывод ffmpeg (stdout): {result.stdout}")
                if result.returncode != 0:
                    logger.error(f"Ошибка ffmpeg при конвертации в WAV: {result.stderr}")
                    raise Exception(f"Ошибка ffmpeg: {result.stderr}")
                
                # Проверяем размер WAV файла
                wav_size = os.path.getsize(wav_file.name)
                logger.debug(f"Размер WAV файла: {wav_size} байт")
                
                logger.info("Конвертация в WAV успешно завершена")
            except Exception as e:
                logger.error(f"Ошибка при конвертации в WAV: {str(e)}")
                logger.exception("Полный стек ошибки при конвертации:")
                await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                return
            
            # Загружаем WAV файл
            try:
                logger.debug(f"Начало чтения WAV файла: {wav_file.name}")
                wav_data, sample_rate = sf.read(wav_file.name)
                logger.debug(f"WAV файл загружен, частота дискретизации: {sample_rate}")
                logger.debug(f"Размер аудио данных: {wav_data.shape}")
                logger.debug(f"Тип данных: {wav_data.dtype}")
                logger.debug(f"Диапазон значений: min={wav_data.min()}, max={wav_data.max()}")
            except Exception as e:
                logger.error(f"Ошибка при чтении WAV файла: {str(e)}")
                logger.exception("Полный стек ошибки при чтении WAV:")
                await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                return
            
            # Применяем эффект
            logger.debug(f"Применение эффекта {effect}")
            try:
                if effect == 'robot':
                    logger.debug("Начало применения эффекта робота")
                    processed_audio, new_sample_rate = apply_robot_effect(wav_data, sample_rate)
                elif effect == 'musical':
                    logger.debug("Начало применения музыкального эффекта голоса")
                    processed_audio, new_sample_rate = apply_musical_voice_effect(wav_data, sample_rate)
                elif effect == 'autotune':
                    logger.debug("Начало применения эффекта автотюна")
                    processed_audio, new_sample_rate = apply_autotune_effect(wav_data, sample_rate)
                elif effect == 'rough':
                    logger.debug("Начало применения эффекта грубого голоса")
                    processed_audio, new_sample_rate = apply_rough_voice_effect(wav_data, sample_rate)
                else:
                    logger.warning(f"Неизвестный эффект {effect} выбран пользователем {user_id}")
                    await query.message.edit_text("Неизвестный эффект.")
                    return
                
                logger.debug(f"Эффект применен, размер обработанных данных: {processed_audio.shape}")
                logger.debug(f"Новая частота дискретизации: {new_sample_rate}")
                logger.debug(f"Диапазон значений после обработки: min={processed_audio.min()}, max={processed_audio.max()}")
                
                logger.info(f"Эффект {effect} применен, новая частота дискретизации: {new_sample_rate}")
            except Exception as e:
                logger.error(f"Ошибка при применении эффекта {effect}: {str(e)}")
                logger.exception("Полный стек ошибки при применении эффекта:")
                await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                return
            
            # Сохраняем обработанный аудио
            with tempfile.NamedTemporaryFile(suffix='.ogg') as output_file:
                try:
                    # Сначала сохраняем как WAV
                    logger.debug(f"Сохранение обработанного аудио в: {output_file.name}")
                    sf.write(output_file.name, processed_audio, new_sample_rate)
                    logger.debug("Обработанный аудио сохранен как WAV")
                    
                    # Проверяем размер файла
                    output_size = os.path.getsize(output_file.name)
                    logger.debug(f"Размер обработанного WAV файла: {output_size} байт")
                    
                except Exception as e:
                    logger.error(f"Ошибка при сохранении WAV: {str(e)}")
                    logger.exception("Полный стек ошибки при сохранении WAV:")
                    await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                    return
                
                # Конвертируем в OGG с помощью ffmpeg
                ogg_file = tempfile.NamedTemporaryFile(suffix='.ogg')
                try:
                    logger.debug("Начало конвертации в OGG с помощью ffmpeg")
                    logger.debug(f"Исходный файл: {output_file.name}")
                    logger.debug(f"Целевой файл: {ogg_file.name}")
                    
                    result = subprocess.run([
                        'ffmpeg', '-y', '-i', output_file.name,
                        '-acodec', 'libopus',
                        '-ar', '48000',
                        '-ac', '1',
                        ogg_file.name
                    ], capture_output=True, text=True)
                    
                    logger.debug(f"Вывод ffmpeg (stdout): {result.stdout}")
                    if result.returncode != 0:
                        logger.error(f"Ошибка ffmpeg при конвертации в OGG: {result.stderr}")
                        raise Exception(f"Ошибка ffmpeg: {result.stderr}")
                    
                    # Проверяем размер OGG файла
                    ogg_size = os.path.getsize(ogg_file.name)
                    logger.debug(f"Размер OGG файла: {ogg_size} байт")
                    
                    logger.info("Конвертация в OGG успешно завершена")
                except Exception as e:
                    logger.error(f"Ошибка при конвертации в OGG: {str(e)}")
                    logger.exception("Полный стек ошибки при конвертации в OGG:")
                    await query.message.edit_text("Ошибка при обработке аудио. Пожалуйста, попробуйте еще раз.")
                    return
                
                # Отправляем обработанное сообщение
                try:
                    logger.debug(f"Отправка обработанного сообщения в чат {voice_info['chat_id']}")
                    logger.debug(f"ID сообщения для ответа: {voice_info['message_id']}")
                    
                    await context.bot.send_voice(
                        chat_id=voice_info['chat_id'],
                        voice=open(ogg_file.name, 'rb'),
                        caption=f"Эффект: {EFFECTS[effect]}",
                        reply_to_message_id=voice_info['message_id']
                    )
                    logger.info(f"Обработанное голосовое сообщение отправлено в чат {voice_info['chat_id']}")
                except Exception as e:
                    logger.error(f"Ошибка при отправке обработанного сообщения: {str(e)}")
                    logger.exception("Полный стек ошибки при отправке:")
                    await query.message.edit_text("Ошибка при отправке обработанного сообщения. Пожалуйста, попробуйте еще раз.")
                    return
                
                # Удаляем сообщение с кнопками
                try:
                    logger.debug("Удаление сообщения с кнопками")
                    await query.message.delete()
                    logger.info("Сообщение с кнопками удалено")
                except Exception as e:
                    logger.warning(f"Не удалось удалить сообщение с кнопками: {str(e)}")
                    logger.exception("Полный стек ошибки при удалении сообщения:")
                
                # Удаляем информацию о голосовом сообщении
                del voice_messages[user_id]
                logger.info(f"Информация о голосовом сообщении удалена для пользователя {user_id}")
                logger.debug(f"Текущее состояние voice_messages: {voice_messages}")
    
    except Exception as e:
        logger.error(f"Ошибка при обработке голосового сообщения для пользователя {user_id}: {str(e)}")
        logger.exception("Полный стек ошибки:")
        await query.message.edit_text("Произошла ошибка при обработке голосового сообщения. Пожалуйста, попробуйте еще раз.")
        # В случае ошибки также удаляем информацию о сообщении
        if user_id in voice_messages:
            del voice_messages[user_id]
            logger.info(f"Информация о голосовом сообщении удалена после ошибки для пользователя {user_id}")
            logger.debug(f"Текущее состояние voice_messages: {voice_messages}")

def apply_robot_effect(audio, sample_rate):
    """Применяет эффект робота с использованием pydub"""
    try:
        logger.info("=== НАЧАЛО ЭФФЕКТА РОБОТА ===")
        
        # Конвертируем numpy array в AudioSegment
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # Добавляем эффект робота
        # 1. Ускоряем аудио для более механического звука
        audio_segment = speedup(audio_segment, playback_speed=1.2)
        
        # 2. Добавляем квадратную волну для металлического звука
        square_wave = Square(440).to_audio_segment(duration=len(audio_segment))
        square_wave = square_wave - 20  # Уменьшаем громкость
        audio_segment = audio_segment.overlay(square_wave)
        
        # 3. Добавляем шум для механического эффекта
        noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))
        noise = noise - 30  # Уменьшаем громкость шума
        audio_segment = audio_segment.overlay(noise)
        
        # 4. Компрессируем динамический диапазон
        audio_segment = compress_dynamic_range(audio_segment, threshold=-20, ratio=4.0)
        
        # 5. Нормализуем
        audio_segment = normalize(audio_segment)
        
        # Конвертируем обратно в numpy array
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='wav')
        buffer.seek(0)
        processed_audio, new_sample_rate = sf.read(buffer)
        
        logger.info("=== ЭФФЕКТ РОБОТА УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, new_sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении эффекта робота: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_musical_voice_effect(audio, sample_rate):
    """Применяет музыкальный эффект голоса с использованием pydub"""
    try:
        logger.info("=== НАЧАЛО МУЗЫКАЛЬНОГО ЭФФЕКТА ГОЛОСА ===")
        
        # Конвертируем numpy array в AudioSegment
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # 1. Добавляем гармоники
        harmonics = []
        for freq in [440, 880, 1320]:  # Основной тон и две гармоники
            sine_wave = Sine(freq).to_audio_segment(duration=len(audio_segment))
            sine_wave = sine_wave - 15  # Уменьшаем громкость
            harmonics.append(sine_wave)
        
        for harmonic in harmonics:
            audio_segment = audio_segment.overlay(harmonic)
        
        # 2. Добавляем легкое вибрато
        vibrato = Sine(5).to_audio_segment(duration=len(audio_segment))
        vibrato = vibrato - 25  # Уменьшаем громкость
        audio_segment = audio_segment.overlay(vibrato)
        
        # 3. Компрессируем динамический диапазон
        audio_segment = compress_dynamic_range(audio_segment, threshold=-25, ratio=3.0)
        
        # 4. Нормализуем
        audio_segment = normalize(audio_segment)
        
        # Конвертируем обратно в numpy array
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='wav')
        buffer.seek(0)
        processed_audio, new_sample_rate = sf.read(buffer)
        
        logger.info("=== МУЗЫКАЛЬНЫЙ ЭФФЕКТ ГОЛОСА УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, new_sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении музыкального эффекта голоса: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_autotune_effect(audio, sample_rate):
    """Применяет эффект автотюна с использованием pydub"""
    try:
        logger.info("=== НАЧАЛО АВТОТЮНА ===")
        
        # Конвертируем numpy array в AudioSegment
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # 1. Определяем базовую ноту
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7')
        )
        
        if len(f0[~np.isnan(f0)]) > 0:
            base_note = librosa.hz_to_midi(np.median(f0[~np.isnan(f0)]))
        else:
            base_note = 60  # C4 по умолчанию
        
        # 2. Применяем коррекцию высоты тона
        audio_segment = pitch_shift(audio_segment, n_steps=base_note-60)
        
        # 3. Добавляем сильное вибрато
        vibrato = Sine(7).to_audio_segment(duration=len(audio_segment))
        vibrato = vibrato - 20  # Уменьшаем громкость
        audio_segment = audio_segment.overlay(vibrato)
        
        # 4. Компрессируем динамический диапазон
        audio_segment = compress_dynamic_range(audio_segment, threshold=-20, ratio=4.0)
        
        # 5. Нормализуем
        audio_segment = normalize(audio_segment)
        
        # Конвертируем обратно в numpy array
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='wav')
        buffer.seek(0)
        processed_audio, new_sample_rate = sf.read(buffer)
        
        logger.info("=== АВТОТЮН УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, new_sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении автотюна: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_rough_voice_effect(audio, sample_rate):
    """Применяет эффект грубого голоса с использованием pydub"""
    try:
        logger.info("=== НАЧАЛО ЭФФЕКТА ГРУБОГО ГОЛОСА ===")
        
        # Конвертируем numpy array в AudioSegment
        audio_segment = AudioSegment(
            audio.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        
        # 1. Добавляем искажение
        audio_segment = audio_segment + 10  # Усиливаем сигнал
        
        # 2. Добавляем шум
        noise = WhiteNoise().to_audio_segment(duration=len(audio_segment))
        noise = noise - 25  # Уменьшаем громкость шума
        audio_segment = audio_segment.overlay(noise)
        
        # 3. Добавляем квадратную волну для грубости
        square_wave = Square(220).to_audio_segment(duration=len(audio_segment))
        square_wave = square_wave - 15  # Уменьшаем громкость
        audio_segment = audio_segment.overlay(square_wave)
        
        # 4. Компрессируем динамический диапазон
        audio_segment = compress_dynamic_range(audio_segment, threshold=-15, ratio=5.0)
        
        # 5. Нормализуем
        audio_segment = normalize(audio_segment)
        
        # Конвертируем обратно в numpy array
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='wav')
        buffer.seek(0)
        processed_audio, new_sample_rate = sf.read(buffer)
        
        logger.info("=== ЭФФЕКТ ГРУБОГО ГОЛОСА УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, new_sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении эффекта грубого голоса: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

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