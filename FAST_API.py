#################################### Book Stroe Prototype ################################
#################################### Babak EA 2024_02_18 ################################


# from fastapi import FastAPI, UploadFile, File, HTTPException, Query,Form
from enum import Enum
import pandas as pd
from pydantic import BaseModel
import io
import logging
from typing import List

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Cookie, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from Witcher.Book_Store import *
import nest_asyncio
import uvicorn




Data_base=Data_Base()
Data_base._create_database_()
Chatbot=Vector_Dataset_Generation(Path="./Data/booksummaries.txt",Local_Data="./Chroma_new/",Log_Databse=Data_base)

app = FastAPI()

# Route for uploading CSV files and API key
@app.post("/Chat/")
async def Data_Registery(
    User_ID: str = Query(..., description="Enter Your ID Please"),
    Question: str = Query(..., description="User Questions "),):

    result = Chatbot._Call_(Question=Question,User_ID=User_ID)
    return result
@app.post("/user_serach/")
async def user_search(
    User_ID: str = Query(..., description="Enter Your ID Please")):
    result=Chatbot.Log_Database.get_user_info(User_ID)
    return result
    
@app.post("/user_registry/")
async def user_registry(
    User_ID: str = Query(..., description="Enter Your ID Please"),
    email_address: str = Query(..., description="Your Email Please"),
    user_phone: str = Query(..., description="Your valid Phone number please ")):
    result=Chatbot.Log_Database.insert_user(User_ID,email_address,user_phone)
    return result
    
@app.post("/user_info/")
async def user_registry(
    User_ID: str = Query(..., description="Enter Your ID Please")):
    result=Chatbot.Log_Database.get_search_hist(User_ID)
    return result

@app.post("/Genres_Recommender/")
async def user_Genre_Recommender(
    User_ID: str = Query(..., description="Enter Your ID Please")):
    result=Chatbot.Recommender_Grnre(User_ID)
    return result

