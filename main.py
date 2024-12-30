import asyncio
import logging
import sys
from os import getenv
from dotenv import load_dotenv

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

from generation import ResponseGenerator

load_dotenv()
TOKEN = getenv("BOT_TOKEN")

dp = Dispatcher()
generator = ResponseGenerator()


@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    answer = f"""
    Hello, {html.bold(message.from_user.full_name)}!
    
I can help you cope with mental problems!
You can ask me how you feel and I will support you.
"""

    await message.answer(answer)


@dp.message()
async def echo_handler(message: Message) -> None:
    """
    Handler will answer user question
    """
    response = generator.process_query(message.text)
    print(message.text)
    print(response)

    await message.answer(response)


async def main() -> None:
    bot = Bot(token=TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())