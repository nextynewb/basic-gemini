import fastapi
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.document_loaders import WikipediaLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain.output_parsers.datetime import DatetimeOutputParser


load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
llm = ChatGoogleGenerativeAI(api_key=GEMINI_API_KEY, model="gemini-1.5-flash")
output_parser = DatetimeOutputParser()

app = FastAPI(title="Gemini API Example",
              description="This is an example of how to use the Gemini API with FastAPI")


class Message(BaseModel):
    message: str
    topic: str


@app.get("/send-message/")
async def send_message(msg: Message):
    return {"message": msg.message}


@app.get("/custom-questions/")
async def custom_endpoint(topic: str, question: str):
    loader = WikipediaLoader(
        query=topic,
        load_max_docs=1
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system",
             "You are a helpful assistant "),
            ("human",
             "This Question: \n {question} \n Here is some extra context: \n {context} and reply in {format_instructions} format")
        ]
    )

    chain = prompt | llm
    ai_msg = chain.invoke({
        "question": question,
        "context": loader.load()[0].page_content,
        "format_instructions": output_parser.get_format_instructions()
    })

    return {"message": ai_msg.content}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
