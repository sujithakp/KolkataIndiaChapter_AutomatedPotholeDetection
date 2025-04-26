import asyncio
import logging
from services.telegram_bot import send_telegram_message_with_photo

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('test_telegram.log')]
)
logger = logging.getLogger(__name__)

async def test_telegram():
    try:
        await send_telegram_message_with_photo("Test message from Pothole Detector")
        logger.info("Test message sent successfully")
    except Exception as e:
        logger.error(f"Test message failed: {str(e)}", exc_info=True)

if __name__ == "__main__":
    asyncio.run(test_telegram())