import os
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

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я бот для модификации голосовых сообщений. "
        "Ответьте на голосовое сообщение, и я предложу варианты его модификации."
    )

async def handle_reply(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message.reply_to_message or not update.message.reply_to_message.voice:
        await update.message.reply_text("Пожалуйста, ответьте на голосовое сообщение.")
        return

    voice = update.message.reply_to_message.voice
    context.user_data['voice_message'] = voice
    context.user_data['reply_to_message_id'] = update.message.reply_to_message.message_id

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
    query = update.callback_query
    await query.answer()

    effect = query.data
    voice = context.user_data.get('voice_message')
    reply_to_message_id = context.user_data.get('reply_to_message_id')

    if not voice or not reply_to_message_id:
        await query.edit_message_text("Извините, произошла ошибка. Попробуйте отправить сообщение снова.")
        return

    await query.edit_message_text("Обрабатываю голосовое сообщение...")

    try:
        voice_file = await context.bot.get_file(voice.file_id)
        await voice_file.download_to_drive("temp.ogg")
        
        audio = AudioSegment.from_ogg("temp.ogg")
        audio.export("temp.wav", format="wav")

        # Здесь будет обработка аудио
        # Пока просто отправляем оригинальное сообщение
        
        with open("temp.wav", "rb") as audio_file:
            await context.bot.send_voice(
                chat_id=update.effective_chat.id,
                voice=audio_file,
                reply_to_message_id=reply_to_message_id
            )

        os.remove("temp.ogg")
        os.remove("temp.wav")
        await query.edit_message_text(f"Эффект {EFFECTS[effect]} применен!")
    except Exception as e:
        await query.edit_message_text(f"Произошла ошибка: {str(e)}")

def main():
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.REPLY & filters.TEXT, handle_reply))
    application.add_handler(CallbackQueryHandler(apply_effect))
    application.run_polling()

if __name__ == '__main__':
    main() 