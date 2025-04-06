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
    'rough': 'Грубый голос',
    'echo': 'Эффект эха',
    'slow': 'Замедление',
    'fast': 'Ускорение',
    'reverse': 'Обратный эффект',
    'autotune': 'Автотюн',
    'autotune2': 'Автотюн 2',  # Добавляем новый эффект
    'autotune3': 'Автотюн 3'  # Добавляем третий эффект
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
                    processed_audio = apply_robot_effect(wav_data, sample_rate)
                    new_sample_rate = sample_rate
                elif effect == 'rough':
                    logger.debug("Начало применения эффекта грубого голоса")
                    processed_audio = apply_rough_voice_effect(wav_data, sample_rate)
                    new_sample_rate = sample_rate
                elif effect == 'echo':
                    logger.debug("Начало применения эффекта эха")
                    processed_audio = apply_echo_effect(wav_data, sample_rate)
                    new_sample_rate = sample_rate
                elif effect == 'slow':
                    logger.debug("Начало применения эффекта замедления")
                    processed_audio, new_sample_rate = apply_slow_effect(wav_data, sample_rate)
                elif effect == 'fast':
                    logger.debug("Начало применения эффекта ускорения")
                    processed_audio, new_sample_rate = apply_fast_effect(wav_data, sample_rate)
                elif effect == 'reverse':
                    logger.debug("Начало применения обратного эффекта")
                    processed_audio, new_sample_rate = apply_reverse_effect(wav_data, sample_rate)
                elif effect == 'autotune':
                    logger.debug("Начало применения эффекта автотюна")
                    processed_audio, new_sample_rate = apply_autotune_effect(wav_data, sample_rate)
                elif effect == 'autotune2':
                    logger.debug("Начало применения эффекта автотюна 2")
                    processed_audio, new_sample_rate = apply_autotune_effect_2(wav_data, sample_rate)
                elif effect == 'autotune3':
                    logger.debug("Начало применения эффекта автотюна 3")
                    processed_audio, new_sample_rate = apply_autotune_effect_3(wav_data, sample_rate)
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
    """Применяет эффект робота к аудио"""
    try:
        logger.info("=== НАЧАЛО ЭФФЕКТА РОБОТА ===")
        
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        logger.info(f"Аудио нормализовано, диапазон: [{np.min(audio)}, {np.max(audio)}]")
        
        # Применяем шумоподавление
        logger.info("Применяем шумоподавление...")
        # Используем медианный фильтр для удаления шума
        audio = scipy.signal.medfilt(audio, kernel_size=5)
        
        # Разбиваем аудио на фреймы
        frame_length = 1024
        hop_length = 256
        logger.info(f"Разбиваем аудио на фреймы: frame_length={frame_length}, hop_length={hop_length}")
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        logger.info(f"Разбито на {frames.shape[1]} фреймов")
        
        # Применяем эффект к каждому фрейму
        logger.info("Начинаем обработку фреймов...")
        processed_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # Применяем быстрое преобразование Фурье
            frame_fft = np.fft.rfft(frame)
            
            # Усиливаем высокие частоты для более "металлического" звука
            freq = np.fft.rfftfreq(len(frame), 1.0/sample_rate)
            high_freq_mask = freq > 1000  # Частоты выше 1 кГц
            frame_fft[high_freq_mask] *= 2.0  # Усиливаем высокие частоты
            
            # Добавляем гармоники для более "роботизированного" звука
            harmonics = np.zeros_like(frame_fft)
            for h in range(2, 5):  # Добавляем 2-ю, 3-ю и 4-ю гармоники
                if h * len(frame_fft) < len(frame_fft):
                    harmonics[:len(frame_fft)//h] += frame_fft[::h] * 0.5
            frame_fft += harmonics
            
            # Обратное преобразование Фурье
            processed_frame = np.fft.irfft(frame_fft)
            
            # Нормализуем фрейм
            processed_frame = processed_frame / np.max(np.abs(processed_frame))
            
            processed_frames.append(processed_frame)
        
        # Собираем обработанные фреймы обратно в аудио
        logger.info("Собираем обработанные фреймы...")
        processed_audio = librosa.util.overlap_and_add(np.array(processed_frames).T, hop_length=hop_length)
        
        # Обрезаем до исходной длины
        processed_audio = processed_audio[:len(audio)]
        
        # Нормализуем
        processed_audio = processed_audio / np.max(np.abs(processed_audio))
        logger.info(f"Нормализованный выход: [{np.min(processed_audio)}, {np.max(processed_audio)}]")
        
        # Добавляем легкую реверберацию
        logger.info("Добавляем легкую реверберацию...")
        processed_audio = librosa.effects.preemphasis(processed_audio, coef=0.97)
        
        logger.info("=== ЭФФЕКТ РОБОТА УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении эффекта робота: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_rough_voice_effect(audio, sample_rate):
    """Применяет эффект грубого голоса с усиленным искажением"""
    try:
        logger.info("=== НАЧАЛО ЭФФЕКТА ГРУБОГО ГОЛОСА ===")
        logger.info(f"Размер входного аудио: {audio.shape}, тип: {audio.dtype}")
        logger.info(f"Частота дискретизации: {sample_rate}")
        
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        logger.info(f"Аудио нормализовано, диапазон: [{np.min(audio)}, {np.max(audio)}]")
        
        # Разбиваем аудио на фреймы
        frame_length = 1024  # Уменьшаем размер фрейма для более резких переходов
        hop_length = 256    # Уменьшаем перекрытие для более грубого звука
        logger.info(f"Разбиваем аудио на фреймы: frame_length={frame_length}, hop_length={hop_length}")
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        logger.info(f"Разбито на {frames.shape[1]} фреймов")
        
        # Применяем эффект к каждому фрейму
        logger.info("Начинаем обработку фреймов...")
        processed_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            # Применяем FFT
            frame_fft = np.fft.rfft(frame)
            freqs = np.fft.rfftfreq(len(frame), 1/sample_rate)
            
            # Усиливаем высокие частоты для более грубого звука
            high_freq_boost = np.linspace(1, 3, len(frame_fft))  # Увеличиваем усиление
            frame_fft = frame_fft * high_freq_boost
            
            # Добавляем гармоники для более агрессивного звука
            harmonics = np.zeros_like(frame_fft)
            for h in range(2, 6):  # Добавляем больше гармоник
                if h * len(frame_fft) < len(frame_fft):
                    harmonics[:len(frame_fft)//h] += 0.3 * frame_fft[::h]  # Усиливаем гармоники
            
            frame_fft = frame_fft + harmonics
            
            # Применяем обратное FFT
            processed_frame = np.fft.irfft(frame_fft)
            
            # Добавляем искажение
            processed_frame = np.tanh(processed_frame * 3)  # Усиливаем искажение
            
            # Добавляем шум
            noise = np.random.normal(0, 0.05, len(processed_frame))  # Увеличиваем уровень шума
            processed_frame = processed_frame + noise
            
            # Нормализуем фрейм
            processed_frame = processed_frame / np.max(np.abs(processed_frame))
            
            processed_frames.append(processed_frame)
        
        # Собираем обработанные фреймы обратно в аудио
        logger.info("Собираем обработанные фреймы...")
        processed_audio = librosa.util.overlap_and_add(np.array(processed_frames).T, hop_length=hop_length)
        
        # Обрезаем до исходной длины
        processed_audio = processed_audio[:len(audio)]
        
        # Добавляем финальное искажение
        processed_audio = np.tanh(processed_audio * 2)  # Финальное искажение
        
        # Нормализуем
        processed_audio = processed_audio / np.max(np.abs(processed_audio))
        logger.info(f"Нормализованный выход: [{np.min(processed_audio)}, {np.max(processed_audio)}]")
        
        # Добавляем реверберацию с коротким временем затухания
        logger.info("Добавляем реверберацию...")
        processed_audio = librosa.effects.preemphasis(processed_audio, coef=0.95)
        
        logger.info("=== ЭФФЕКТ ГРУБОГО ГОЛОСА УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении эффекта грубого голоса: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

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

def apply_autotune_effect(audio, sample_rate):
    """Применение эффекта автотюна с помощью pyworld"""
    logger.debug("=== НАЧАЛО АВТОТЮНА ===")
    logger.debug(f"Размер входного аудио: {audio.shape}, тип: {audio.dtype}")
    logger.debug(f"Частота дискретизации: {sample_rate}")
    logger.debug(f"Диапазон входного аудио: [{np.min(audio)}, {np.max(audio)}]")
    
    try:
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        logger.debug(f"Аудио нормализовано, диапазон: [{np.min(audio)}, {np.max(audio)}]")
        
        # Извлекаем основные параметры голоса
        logger.debug("Извлечение параметров голоса...")
        _f0, t = pw.dio(audio.astype(np.float64), sample_rate)  # Основная частота
        logger.debug(f"Первичная f0: мин={np.min(_f0[_f0 > 0])}, макс={np.max(_f0)}")
        
        f0 = pw.stonemask(audio.astype(np.float64), _f0, t, sample_rate)  # Уточнение частоты
        logger.debug(f"Уточненная f0: мин={np.min(f0[f0 > 0])}, макс={np.max(f0)}")
        
        sp = pw.cheaptrick(audio.astype(np.float64), f0, t, sample_rate)  # Спектральная оболочка
        ap = pw.d4c(audio.astype(np.float64), f0, t, sample_rate)  # Апериодичность
        
        logger.debug(f"Размеры: f0={f0.shape}, sp={sp.shape}, ap={ap.shape}")
        
        # Определяем базовую ноту (тональность)
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) > 0:
            base_note = librosa.hz_to_midi(np.median(valid_f0))
            logger.debug(f"Найдена базовая нота: {base_note}")
            # Округляем до ближайшей ноты в мажорной гамме
            scale_notes = [0, 2, 4, 5, 7, 9, 11]  # Мажорная гамма
            base_note = np.round(base_note)
            base_note = base_note - (base_note % 12)  # Округляем до октавы
            logger.debug(f"Округленная базовая нота: {base_note}")
        else:
            base_note = 60  # C4 как базовая нота по умолчанию
            logger.debug("Используется базовая нота по умолчанию: C4")
        
        # Квантуем высоту тона в ноты мажорной гаммы
        logger.debug("Квантование высоты тона в мажорную гамму...")
        notes = librosa.hz_to_midi(f0)
        quantized_notes = np.zeros_like(notes)
        
        for i in range(len(notes)):
            if notes[i] > 0:  # Если есть тон
                # Находим ближайшую ноту в гамме
                note_in_scale = notes[i] - base_note
                octave = np.floor(note_in_scale / 12)
                note_in_octave = note_in_scale % 12
                
                # Находим ближайшую ноту в гамме
                distances = [abs(note_in_octave - scale_note) for scale_note in scale_notes]
                closest_scale_note = scale_notes[np.argmin(distances)]
                
                # Квантуем в ближайшую ноту гаммы
                quantized_notes[i] = base_note + octave * 12 + closest_scale_note
                logger.debug(f"Исходная нота: {notes[i]}, квантованная: {quantized_notes[i]}")
            else:
                quantized_notes[i] = 0
        
        # Преобразуем квантованные ноты обратно в частоты
        quantized_f0 = librosa.midi_to_hz(quantized_notes)
        logger.debug(f"Квантованные частоты: мин={np.min(quantized_f0[quantized_f0 > 0])}, макс={np.max(quantized_f0)}")
        
        # Усиливаем эффект автотюна
        logger.debug("Усиление эффекта автотюна...")
        # 1. Увеличиваем силу коррекции
        n_steps = librosa.hz_to_midi(quantized_f0) - librosa.hz_to_midi(f0)
        n_steps = np.nan_to_num(n_steps, nan=0)
        n_steps = n_steps * 2.0  # Усиливаем на 100%
        logger.debug(f"Коррекция высоты тона: мин={np.min(n_steps)}, макс={np.max(n_steps)}")
        
        # 2. Добавляем сильное вибрато
        t = np.arange(len(f0))
        vibrato = 0.5 * np.sin(2 * np.pi * 5 * t / len(f0))  # Усиленное вибрато
        n_steps = n_steps + vibrato
        logger.debug(f"Добавлено вибрато: амплитуда=0.5, частота=5 Гц")
        
        # 3. Ограничиваем максимальное изменение
        n_steps = np.clip(n_steps, -36, 36)  # Три октавы вверх или вниз
        logger.debug(f"Ограниченная коррекция: мин={np.min(n_steps)}, макс={np.max(n_steps)}")
        
        # Применяем изменение высоты тона
        logger.debug("Применение изменения высоты тона...")
        new_f0 = librosa.midi_to_hz(librosa.hz_to_midi(f0) + n_steps)
        logger.debug(f"Новые частоты: мин={np.min(new_f0[new_f0 > 0])}, макс={np.max(new_f0)}")
        
        # Синтезируем новый звук
        logger.debug("Синтез нового звука...")
        processed_audio = pw.synthesize(new_f0, sp, ap, sample_rate)
        
        # Нормализуем выходной сигнал
        processed_audio = processed_audio / np.max(np.abs(processed_audio))
        logger.debug(f"Нормализованный выход: мин={np.min(processed_audio)}, макс={np.max(processed_audio)}")
        
        # Добавляем сильную реверберацию
        processed_audio = librosa.effects.preemphasis(processed_audio, coef=0.97)
        logger.debug("Добавлена реверберация")
        
        logger.debug("=== АВТОТЮН УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении автотюна: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_autotune_effect_2(audio, sample_rate):
    """Альтернативный вариант автотюна - максимально заметный эффект"""
    # Импортируем необходимые библиотеки
    import numpy as np
    import librosa
    import soundfile as sf
    
    logger.info("=== НАЧАЛО АВТОТЮНА 2 (МАКСИМАЛЬНЫЙ ЭФФЕКТ) ===")
    logger.info(f"Размер входного аудио: {audio.shape}, тип: {audio.dtype}")
    logger.info(f"Частота дискретизации: {sample_rate}")
    logger.info(f"Диапазон входного аудио: [{np.min(audio)}, {np.max(audio)}]")
    
    try:
        # Проверяем наличие необходимых библиотек
        logger.info("Проверка наличия библиотек...")
        logger.info("Все необходимые библиотеки импортированы успешно")
        
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        logger.info(f"Аудио нормализовано, диапазон: [{np.min(audio)}, {np.max(audio)}]")
        
        # Разбиваем аудио на фреймы
        frame_length = 1024  # Уменьшаем размер фрейма для более быстрой реакции
        hop_length = 256    # Уменьшаем перекрытие для более частой коррекции
        logger.info(f"Разбиваем аудио на фреймы: frame_length={frame_length}, hop_length={hop_length}")
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        logger.info(f"Разбито на {frames.shape[1]} фреймов")
        
        # Определяем базовую ноту
        logger.info("Определяем базовую ноту с помощью librosa.pyin...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=frame_length,
            hop_length=hop_length
        )
        logger.info(f"Определена базовая частота: {np.median(f0[~np.isnan(f0)])} Гц")
        logger.info(f"Количество фреймов с голосом: {np.sum(voiced_flag)} из {len(voiced_flag)}")
        
        # Определяем тональность
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            base_note = librosa.hz_to_midi(np.median(valid_f0))
            logger.info(f"Найдена базовая нота: {base_note}")
            # Округляем до ближайшей ноты в мажорной гамме
            scale_notes = [0, 2, 4, 5, 7, 9, 11]  # Мажорная гамма
            base_note = np.round(base_note)
            base_note = base_note - (base_note % 12)  # Округляем до октавы
            logger.info(f"Округленная базовая нота: {base_note}")
        else:
            base_note = 60  # C4 как базовая нота по умолчанию
            logger.warning("Используется базовая нота по умолчанию: C4")
        
        # Применяем автотюн к каждому фрейму
        logger.info("Начинаем обработку фреймов...")
        processed_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            if voiced_flag[i]:  # Если фрейм содержит голос
                # Определяем текущую ноту
                current_note = librosa.hz_to_midi(f0[i])
                logger.debug(f"Фрейм {i}: текущая нота = {current_note}")
                
                # Находим ближайшую ноту в гамме
                note_in_scale = current_note - base_note
                octave = np.floor(note_in_scale / 12)
                note_in_octave = note_in_scale % 12
                
                # Находим ближайшую ноту в гамме
                distances = [abs(note_in_octave - scale_note) for scale_note in scale_notes]
                closest_scale_note = scale_notes[np.argmin(distances)]
                
                # Квантуем в ближайшую ноту гаммы
                target_note = base_note + octave * 12 + closest_scale_note
                logger.debug(f"Фрейм {i}: целевая нота = {target_note}")
                
                # Вычисляем необходимое изменение высоты тона
                n_steps = target_note - current_note
                
                # Усиливаем эффект в 3 раза
                n_steps = n_steps * 3.0
                
                # Добавляем сильное вибрато
                vibrato = 1.0 * np.sin(2 * np.pi * 7 * i / frames.shape[1])  # Увеличили амплитуду и частоту
                n_steps = n_steps + vibrato
                
                # Ограничиваем максимальное изменение
                n_steps = np.clip(n_steps, -36, 36)  # Увеличили диапазон до 3 октав
                logger.debug(f"Фрейм {i}: изменение высоты тона = {n_steps}")
                
                # Применяем изменение высоты тона
                processed_frame = librosa.effects.pitch_shift(
                    frame,
                    sr=sample_rate,
                    n_steps=n_steps,
                    bins_per_octave=48  # Увеличили разрешение
                )
            else:
                processed_frame = frame
                logger.debug(f"Фрейм {i}: пропущен (нет голоса)")
            
            processed_frames.append(processed_frame)
        
        # Собираем обработанные фреймы обратно в аудио
        logger.info("Собираем обработанные фреймы...")
        processed_audio = librosa.util.overlap_and_add(np.array(processed_frames).T, hop_length=hop_length)
        
        # Обрезаем до исходной длины
        processed_audio = processed_audio[:len(audio)]
        
        # Нормализуем
        processed_audio = processed_audio / np.max(np.abs(processed_audio))
        logger.info(f"Нормализованный выход: [{np.min(processed_audio)}, {np.max(processed_audio)}]")
        
        # Добавляем сильную реверберацию
        logger.info("Добавляем сильную реверберацию...")
        processed_audio = librosa.effects.preemphasis(processed_audio, coef=0.99)  # Усилили предыскажение
        
        # Добавляем дополнительное усиление эффекта
        processed_audio = librosa.effects.pitch_shift(
            processed_audio,
            sr=sample_rate,
            n_steps=2,  # Небольшой дополнительный сдвиг
            bins_per_octave=48
        )
        
        logger.info("=== АВТОТЮН 2 (МАКСИМАЛЬНЫЙ ЭФФЕКТ) УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении автотюна 2: {str(e)}")
        logger.exception("Полный стек ошибки:")
        return audio, sample_rate

def apply_autotune_effect_3(audio, sample_rate):
    """Третий вариант автотюна - музыкальная обработка с использованием pydub"""
    try:
        logger.info("=== НАЧАЛО АВТОТЮНА 3 (МУЗЫКАЛЬНЫЙ) ===")
        logger.info(f"Размер входного аудио: {audio.shape}, тип: {audio.dtype}")
        logger.info(f"Частота дискретизации: {sample_rate}")
        
        # Нормализуем аудио
        audio = audio / np.max(np.abs(audio))
        logger.info(f"Аудио нормализовано, диапазон: [{np.min(audio)}, {np.max(audio)}]")
        
        # Разбиваем аудио на фреймы
        frame_length = 2048
        hop_length = 512
        logger.info(f"Разбиваем аудио на фреймы: frame_length={frame_length}, hop_length={hop_length}")
        frames = librosa.util.frame(audio, frame_length=frame_length, hop_length=hop_length)
        logger.info(f"Разбито на {frames.shape[1]} фреймов")
        
        # Определяем базовую ноту и тональность
        logger.info("Определяем базовую ноту и тональность...")
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            frame_length=frame_length,
            hop_length=hop_length
        )
        
        # Определяем тональность
        valid_f0 = f0[~np.isnan(f0)]
        if len(valid_f0) > 0:
            base_note = librosa.hz_to_midi(np.median(valid_f0))
            logger.info(f"Найдена базовая нота: {base_note}")
            
            # Определяем тональность
            chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
            key = np.argmax(np.sum(chroma, axis=1))
            logger.info(f"Определена тональность: {librosa.midi_to_note(key)}")
            
            # Выбираем гамму в зависимости от тональности
            if key in [0, 2, 4, 5, 7, 9, 11]:  # Мажорные тональности
                scale_notes = [0, 2, 4, 5, 7, 9, 11]  # Мажорная гамма
            else:  # Минорные тональности
                scale_notes = [0, 2, 3, 5, 7, 8, 10]  # Минорная гамма
        else:
            base_note = 60  # C4 как базовая нота по умолчанию
            scale_notes = [0, 2, 4, 5, 7, 9, 11]  # Мажорная гамма по умолчанию
            logger.warning("Используется базовая нота по умолчанию: C4")
        
        # Применяем автотюн к каждому фрейму
        logger.info("Начинаем обработку фреймов...")
        processed_frames = []
        for i in range(frames.shape[1]):
            frame = frames[:, i]
            
            if voiced_flag[i]:  # Если фрейм содержит голос
                # Определяем текущую ноту
                current_note = librosa.hz_to_midi(f0[i])
                logger.debug(f"Фрейм {i}: текущая нота = {current_note}")
                
                # Находим ближайшую ноту в гамме
                note_in_scale = current_note - base_note
                octave = np.floor(note_in_scale / 12)
                note_in_octave = note_in_scale % 12
                
                # Находим ближайшую ноту в гамме
                distances = [abs(note_in_octave - scale_note) for scale_note in scale_notes]
                closest_scale_note = scale_notes[np.argmin(distances)]
                
                # Квантуем в ближайшую ноту гаммы
                target_note = base_note + octave * 12 + closest_scale_note
                logger.debug(f"Фрейм {i}: целевая нота = {target_note}")
                
                # Вычисляем необходимое изменение высоты тона
                n_steps = target_note - current_note
                
                # Добавляем музыкальное вибрато
                vibrato = 0.3 * np.sin(2 * np.pi * 5 * i / frames.shape[1])
                n_steps = n_steps + vibrato
                
                # Ограничиваем максимальное изменение
                n_steps = np.clip(n_steps, -12, 12)  # Одна октава вверх или вниз
                logger.debug(f"Фрейм {i}: изменение высоты тона = {n_steps}")
                
                # Применяем изменение высоты тона
                processed_frame = librosa.effects.pitch_shift(
                    frame,
                    sr=sample_rate,
                    n_steps=n_steps,
                    bins_per_octave=24
                )
            else:
                processed_frame = frame
                logger.debug(f"Фрейм {i}: пропущен (нет голоса)")
            
            processed_frames.append(processed_frame)
        
        # Собираем обработанные фреймы обратно в аудио
        logger.info("Собираем обработанные фреймы...")
        processed_audio = librosa.util.overlap_and_add(np.array(processed_frames).T, hop_length=hop_length)
        
        # Обрезаем до исходной длины
        processed_audio = processed_audio[:len(audio)]
        
        # Нормализуем
        processed_audio = processed_audio / np.max(np.abs(processed_audio))
        logger.info(f"Нормализованный выход: [{np.min(processed_audio)}, {np.max(processed_audio)}]")
        
        # Добавляем музыкальную реверберацию
        logger.info("Добавляем музыкальную реверберацию...")
        processed_audio = librosa.effects.preemphasis(processed_audio, coef=0.97)
        
        logger.info("=== АВТОТЮН 3 (МУЗЫКАЛЬНЫЙ) УСПЕШНО ЗАВЕРШЕН ===")
        return processed_audio, sample_rate
        
    except Exception as e:
        logger.error(f"Ошибка при применении автотюна 3: {str(e)}")
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