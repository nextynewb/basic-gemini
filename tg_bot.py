import os
from dotenv import load_dotenv
import telebot
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WikipediaLoader

load_dotenv()

TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_API_KEY')
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-flash")
loader = WikipediaLoader(
    query="Olympic Games",
    load_max_docs=1
)

context_text = loader.load()[0].page_content
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant "),
        ("human",
         "This Question: \n {question} \n Here is some extra context: \n {context}")
    ]
)
bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)


def start(message):
    bot.reply_to(
        message, "Welcome to TayarPro Bot!")


def delete_message(message):
    bot.delete_message(message.chat.id, message.message_id)


def reply_message(message, reply_message):
    bot.reply_to(message, reply_message)

def handle_text_messages(message):

    text = message.text
    chain = prompt | llm

    ai_msg = chain.invoke({
        "question": text,
        "context": context_text,
    })

    reply_message(message, ai_msg.content)


# Register command handler
bot.message_handler(commands=['start'])(start)
bot.message_handler(content_types=['text'])(handle_text_messages)
print("Bot is running....")
bot.polling(none_stop=True)
