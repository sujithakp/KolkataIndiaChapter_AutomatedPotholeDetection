from aiogram import Bot
from aiogram.types import FSInputFile  # Changed from InputFile to FSInputFile
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('telegram_bot.log')
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# Validate environment variables
logger.debug(f"Loaded TELEGRAM_BOT_TOKEN: {TOKEN[:5]}..." if TOKEN else "TELEGRAM_BOT_TOKEN not set")
logger.debug(f"Loaded TELEGRAM_CHAT_ID: {CHAT_ID}" if CHAT_ID else "TELEGRAM_CHAT_ID not set")
if not TOKEN:
    logger.error("TELEGRAM_BOT_TOKEN is not set in .env file")
    raise ValueError("TELEGRAM_BOT_TOKEN is not set")
if not CHAT_ID:
    logger.error("TELEGRAM_CHAT_ID is not set in .env file")
    raise ValueError("TELEGRAM_CHAT_ID is not set")

async def send_telegram_message_with_photo(report_text, photo_path=None):
    """
    Send a Telegram message with text and optional photo.
    
    Args:
        report_text (str): Text to send
        photo_path (str, optional): Path to image file
    """
    MAX_LENGTH = 4096  # Telegram message character limit
    logger.info("Attempting to send Telegram message")
    
    # Verify photo path exists if provided
    if photo_path and not os.path.exists(photo_path):
        logger.error(f"Photo path does not exist: {photo_path}")
        photo_path = None
    
    try:
        bot = Bot(token=TOKEN)
        
        try:
            logger.debug(f"Sending to chat_id: {CHAT_ID}")
            if photo_path and os.path.exists(photo_path):
                logger.info(f"Sending photo from: {photo_path}")
                # Use FSInputFile instead of InputFile for local files
                photo = FSInputFile(photo_path)
                await bot.send_photo(
                    chat_id=CHAT_ID, 
                    photo=photo, 
                    caption=report_text[:1024]  # Caption has a 1024 char limit
                )
            else:
                logger.info("Sending text message without photo")
                await bot.send_message(
                    chat_id=CHAT_ID, 
                    text=report_text[:MAX_LENGTH]
                )
            logger.info("Telegram message sent successfully")
            return True
        finally:
            # Proper cleanup
            await bot.session.close()
            
    except Exception as e:
        logger.error(f"Failed to send Telegram message: {str(e)}", exc_info=True)
        return False