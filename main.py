import os
import asyncio
import tempfile
import wave
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters, ContextTypes
from faster_whisper import WhisperModel
import edge_tts
import aiohttp
import speech_recognition as sr
import subprocess
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get environment variables
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BOT_PASSWORD = os.getenv("BOT_PASSWORD")

# Initialize WhisperModel for STT
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

# OpenAI API settings
API_URL = "https://api.openai.com/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {OPENAI_API_KEY}",
    "Content-Type": "application/json"
}

# TTS settings
TTS_VOICE = "en-GB-RyanNeural"
TTS_RATE = "-10%"
TTS_PITCH = "-10Hz"

# Add this to store conversation history
MAX_HISTORY = 10
conversation_history = {}

# Add this set to store authenticated users
authenticated_users = set()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Welcome! This bot requires authentication. Please use the /auth command followed by the password to start using the bot.")

async def auth(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not context.args:
        await update.message.reply_text("Please provide the password after the /auth command.")
        return
    
    password = context.args[0]
    if password == BOT_PASSWORD:
        authenticated_users.add(update.effective_user.id)
        await update.message.reply_text("Authentication successful. You can now use the bot.")
    else:
        await update.message.reply_text("Incorrect password. Please try again.")

async def clear(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authenticated(update.effective_user.id):
        await update.message.reply_text("You are not authenticated. Please use the /auth command with the correct password.")
        return
    
    user_id = update.effective_user.id
    if user_id in conversation_history:
        conversation_history[user_id] = []
    await update.message.reply_text("Conversation history has been cleared.")

def is_authenticated(user_id: int) -> bool:
    return user_id in authenticated_users

async def handle_voice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authenticated(update.effective_user.id):
        await update.message.reply_text("You are not authenticated. Please use the /auth command with the correct password.")
        return

    try:
        voice = await update.message.voice.get_file()
        with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as voice_file:
            await voice.download_to_drive(voice_file.name)
            
            # Convert ogg to wav using ffmpeg
            wav_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            wav_file.close()  # Close the file before passing it to ffmpeg
            subprocess.run(['ffmpeg', '-i', voice_file.name, '-acodec', 'pcm_s16le', '-ar', '16000', wav_file.name, '-y'], check=True)
            
            # Transcribe audio
            segments, _ = whisper_model.transcribe(wav_file.name, beam_size=5)
            transcript = " ".join(segment.text for segment in segments)

        os.unlink(voice_file.name)
        os.unlink(wav_file.name)

        # Generate AI response
        ai_response = await generate_ai_response(update.effective_user.id, transcript)
        
        # Generate TTS audio
        tts_file = await generate_tts(ai_response)
        
        # Send text and audio response
        await update.message.reply_text(ai_response)
        with open(tts_file, 'rb') as audio:
            await update.message.reply_voice(voice=audio)
        os.unlink(tts_file)

        # Store the text response in user_data for potential later retrieval
        context.user_data['last_response'] = ai_response

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await update.message.reply_text(error_message)
        error_audio = await generate_tts(error_message)
        with open(error_audio, 'rb') as audio:
            await update.message.reply_voice(voice=audio)
        os.unlink(error_audio)

async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not is_authenticated(update.effective_user.id):
        await update.message.reply_text("You are not authenticated. Please use the /auth command with the correct password.")
        return

    try:
        user_input = update.message.text
        
        # Generate AI response
        ai_response = await generate_ai_response(update.effective_user.id, user_input)
        
        # Generate TTS audio
        tts_file = await generate_tts(ai_response)
        
        # Send text and audio response
        await update.message.reply_text(ai_response)
        with open(tts_file, 'rb') as audio:
            await update.message.reply_voice(voice=audio)
        os.unlink(tts_file)

        # Store the text response in user_data for potential later retrieval
        context.user_data['last_response'] = ai_response

    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        await update.message.reply_text(error_message)
        error_audio = await generate_tts(error_message)
        with open(error_audio, 'rb') as audio:
            await update.message.reply_voice(voice=audio)
        os.unlink(error_audio)

async def generate_ai_response(user_id: int, user_input: str) -> str:
    if user_id not in conversation_history:
        conversation_history[user_id] = []
    
    conversation_history[user_id].append({"role": "user", "content": user_input})
    
    # Limit conversation history to last 10 messages
    conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]
    
    messages = [
        {"role": "system", "content": "You are Jarvis, the AI personal assistant from the Marvel Universe, serving Tony Stark. Your task is to assist your user, Sean, with anything they require. Respond as Jarvis would, with a British accent, addressing them as 'Sir,' and providing intelligent, precise assistance in a conversational style. Avoid using bullet points, bold, or italic formatting. Keep your communication fluid, clear, and free from unnecessary embellishments. Do not bolt any sentences or words, just regular plain text, like paragraph format, conversational format, podcast format, without any visual formatting."}
    ] + conversation_history[user_id]

    async with aiohttp.ClientSession() as session:
        async with session.post(API_URL, headers=HEADERS, json={
            "model": "gpt-4",  # or "gpt-3.5-turbo" if you don't have access to GPT-4
            "messages": messages
        }) as response:
            result = await response.json()
            if 'choices' in result and len(result['choices']) > 0:
                ai_response = result['choices'][0]['message']['content']
                conversation_history[user_id].append({"role": "assistant", "content": ai_response})
                return ai_response
            else:
                return f"Error: {result.get('error', {}).get('message', 'Unknown error occurred')}"

async def generate_tts(text: str) -> str:
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate=TTS_RATE, pitch=TTS_PITCH)
    audio_file = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    await communicate.save(audio_file)
    return audio_file

def main() -> None:
    if not all([TELEGRAM_BOT_TOKEN, OPENAI_API_KEY, BOT_PASSWORD]):
        print("Error: Missing environment variables. Please check your .env file.")
        return

    application = ApplicationBuilder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("auth", auth))
    application.add_handler(CommandHandler("clear", clear))
    application.add_handler(MessageHandler(filters.VOICE, handle_voice))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    application.run_polling()

if __name__ == '__main__':
    main()