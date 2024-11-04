import os
from dotenv import load_dotenv
import telebot
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.document_loaders import TextLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA


load_dotenv()

TELEGRAM_BOT_TOKEN = '7307149259:AAGLm7k5PwOi_ggOKm6Ht-GhzKn0wH09XCs'
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


loader = TextLoader("knowledge.txt")
documents = loader.load()
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

# Create a vector
vectorstore = FAISS.from_documents(documents, embeddings)


llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-flash")
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever()
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
    ai_msg = rag_chain.run(text)

    reply_message(message, ai_msg)


# Register command handler
bot.message_handler(commands=['start'])(start)
bot.message_handler(content_types=['text'])(handle_text_messages)
print("Bot is running....")
bot.polling(none_stop=True)
