###############################################3
######## Babak EA  ############################
######## 2024_02_18############################
######## Book STroe Using RAG, Chroma DB ( vecotor database) , OPEN AI ( Summarizer, NER, Intentenion Detection), Fast-API 

########################################################################################################


import random

import os
import pandas as pd
import numpy as np
import re
import chromadb

from langchain.vectorstores import Chroma
from langchain.schema import Document
#embedding Data to chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration

import json
from datetime import datetime
import langchain
#from langchain.document_loaders import PyMuPDFLoade
import sqlite3
import os
import openai
from openai import OpenAI
from langchain import hub
from langchain.chat_models import ChatOpenAI
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.schema import Document
#embedding Data to chromadb
from langchain.embeddings import HuggingFaceBgeEmbeddings

from langchain.embeddings import CacheBackedEmbeddings, HuggingFaceEmbeddings
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

import os
import sys
import json
from datetime import datetime

import langchain
from langchain.document_loaders import PyMuPDFLoader
from fuzzywuzzy import fuzz


############################################################
        
        
class Data_Base:
    
    """
    awesome , 
Now we need a function that insert data to the database, 

like we are passing a JSON file as an input. 
json has these filed, 

user_id, book_id,book_title,genres,author,user_searched,
    
    """

    def __init__(self,Database_Location="./Search_Hist/",Database_Name="Search_host.db"):
        self._Database_loadation=f"{Database_Location}{Database_Name}"
        
    def _create_database_(self):
        # Check if the database file exists
#         if not os.path.exists(self._Database_loadation):
#             # If the database doesn't exist, create it and execute the SQL queries
        conn = sqlite3.connect(self._Database_loadation)
        cursor = conn.cursor()

        # SQL queries to create tables
        create_user_table_query = """
        CREATE TABLE IF NOT EXISTS user_table (
            user_id TEXT PRIMARY KEY,
            email_address TEXT NOT NULL,
            user_phone TEXT
        );
        """

        create_book_search_query = """
        CREATE TABLE IF NOT EXISTS book_search (
            purchase_date DATE NOT NULL,
            user_id TEXT NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT,
            book_genre TEXT,
            bought BOOLEAN NOT NULL
        
        );
        """

        create_book_user_bonous_query = """
        CREATE TABLE IF NOT EXISTS user_bonus (
            user_id TEXT NOT NULL,
            purchase_date DATE NOT NULL,
            Credite int NOT NULL
        );
        """
        create_book_search_hist = """
        CREATE TABLE IF NOT EXISTS search_hist (
            user_id TEXT NOT NULL,
            purchase_date DATE NOT NULL,
            book_title TEXT NOT NULL,
            search text NOT NULL,
            genre TEXT
        );
        """

        # Execute the SQL queries
        cursor.execute(create_user_table_query)
        cursor.execute(create_book_search_query)
        cursor.execute(create_book_user_bonous_query)
        cursor.execute(create_book_search_hist)

        # Commit changes and close the connection
        conn.commit()
        conn.close()


        
    def get_user_info(self,user_id):
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        c.execute("SELECT email_address, user_phone FROM user_table WHERE user_id = ?", (user_id,))
        result = c.fetchone()
        self.conn.close()
        if result:
            return result
        else:
            return None
    
    def insert_user(self,user_id:str, email_address:str, user_phone:str):
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        try:
            c.execute("INSERT INTO user_table (user_id, email_address, user_phone) VALUES (?, ?, ?)", (user_id, email_address, user_phone))
            self.conn.commit()
            self.conn.close()
            return True
        except sqlite3.Error as e:
            print("Error inserting user:", e)
            self.conn.rollback()
            self.conn.close()
            return False
        
        
    ###############################################################################################################    
    def insert_search_hist(self,user_id:str, book_title:str, search:str,genre:str ):
        purchase_date = datetime.now().strftime('%Y-%m-%d')
        
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        """
        user_id TEXT NOT NULL,
            purchase_date DATE NOT NULL,
            book_title TEXT NOT NULL,
            search text NOT NULL,
            genre TEXT
        
        """
        try:
            c.execute("INSERT INTO search_hist (user_id, purchase_date, book_title,search,genre) VALUES (?, ?, ?, ?, ?)",
                                              (user_id, purchase_date, book_title,search,genre))
            self.conn.commit()
            self.conn.close()
            return True
        except sqlite3.Error as e:
            print("Error inserting user:", e)
            self.conn.rollback()
            self.conn.close()
            return False
     
    def get_search_hist(self,user_id):
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        c.execute("SELECT user_id, purchase_date, book_title,search,genre FROM search_hist WHERE user_id = ?", (user_id,))
        result = c.fetchall()
        self.conn.close()
        if result:
            return result
        else:
            return None    
       #######################################################################################################3
    
    def insert_user_bonus(self,user_id:str,Credite:int ):
        purchase_date = datetime.now().strftime('%Y-%m-%d')
        
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        """
            user_id TEXT NOT NULL,
            purchase_date DATE NOT NULL,
            Credite int NOT NULL
        """
        try:
            c.execute("INSERT INTO user_bonus (user_id, purchase_date, Credite) VALUES (?, ?, ?)",
                                              (user_id, purchase_date, Credite))
            self.conn.commit()
            self.conn.close()
            return True
        except sqlite3.Error as e:
            print("Error inserting user:", e)
            self.conn.rollback()
            self.conn.close()
            return False
        
    def get_user_bonus(self,user_id):
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        c.execute("SELECT user_id, purchase_date, Credite FROM user_bonus WHERE user_id = ?", (user_id,))
        result = c.fetchall()
        self.conn.close()
        if result:
            return result
        else:
            return None    
       #######################################################################################################3    
        
        
    def insert_search_book(self,user_id:str,book_title:str,book_author:str,book_genre:str,bought:bool ):
        purchase_date = datetime.now().strftime('%Y-%m-%d')
        
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        """
            purchase_date DATE NOT NULL,
            user_id TEXT NOT NULL,
            book_title TEXT NOT NULL,
            book_author TEXT,
            book_genre TEXT,
            bought BOOLEAN NOT NULL
        """
        try:
            c.execute("INSERT INTO book_search (purchase_date, user_id, book_title,book_author,book_genre,bought) VALUES (?, ?, ?, ?, ?, ?)",
                                              (purchase_date,user_id, book_title, book_author,book_genre,bought))
            self.conn.commit()
            self.conn.close()
            return True
        except sqlite3.Error as e:
            print("Error inserting user:", e)
            self.conn.rollback()
            self.conn.close()
            return False
    def get_search_book(self,user_id):
        self.conn = sqlite3.connect(self._Database_loadation)
        c = self.conn.cursor()
        c.execute("SELECT purchase_date, user_id, book_title,book_author,book_genre,bought FROM book_search WHERE user_id = ?", (user_id,))
        result = c.fetchall()
        self.conn.close()
        if result:
            return result
        else:
            return None  
    ########################################################################################################3
    
    

class Vector_Dataset_Generation:
    def __init__(self,Path:str,Local_Data:str,Log_Databse):
        
        self.Log_Database=Log_Databse
        self._path=Path
        self._local_data=Local_Data
        self.client = chromadb.PersistentClient(path=self._local_data)
        self.collection = self.client.get_or_create_collection(name="book_stroe")
        self.General_collection=self.client.get_or_create_collection(name="general")
        self.user_hist_collection = self.client.get_or_create_collection(name="user_hist") 
        #self.chroma = Chroma()
        self._path_generetore_()
        self._embedding_model="sentence-transformers/all-mpnet-base-v2"
        self.Data=[]
        self.user={}
        self.user_cart={}
        self.Total_tokens=0
        self.Total_cost=0
        
        self._secret_()
        
    def _secret_(self,Path="./secret/",Keyword="secret"):
        STR_Secret=[x for x in os.listdir(f"./{Path}/") if Keyword in x.lower()][0]
        with open(f'./{Path}/{STR_Secret}') as config_file:
            config_data = json.load(config_file)
            self.api_key = config_data.get("api_key")
        
    def _path_generetore_(self):## Generate the Paths (1)
        isExist = os.path.exists(self._local_data)
        if not isExist:
            # Create a new directory because it does not exist
            os.makedirs(self._local_data)
        
    def clean_genres(self,genres_str):
        try:
            genres_dict = json.loads(genres_str)
            genres_list = list(genres_dict.values())
            cleaned_genres = ', '.join(genres_list)
            return cleaned_genres
        except:
            return None
    def _csv_reader_(self):
        self.book_summary_df = pd.read_csv(self._path, 
                              header=None,sep="\t", 
                              names=["Wikipedia ID", "Freebase ID", "Book Title", "Book Author", "Pub date","Genres","Summary"])
        self.book_summary_df['Genres'] = self.book_summary_df['Genres'].apply(self.clean_genres)
        self.book_summary_df=self.book_summary_df[["Wikipedia ID", "Book Title", "Book Author", "Pub date","Genres","Summary"]]
        self.book_summary_df.replace({pd.NA: "",
             "Nan": "",
             "nan": "",
             "": ""
               }, inplace=True)
    
    def _Chroma_Generator_(self):
        documents=self.book_summary_df["Summary"].tolist()
        #ids=self.book_summary_df["Wikipedia ID"].tolist()
        ids=[]
        metadata=[]
        for _, row in self.book_summary_df.iterrows():
            metadata.append( {
                'title': row['Book Title'] if pd.notna(row['Book Title']) else ".",
                'author': row['Book Author'] if pd.notna(row['Book Author']) else ".",
                'pub_date': row['Pub date'] if pd.notna(row['Pub date']) else ".",
                'genres': row['Genres'] if pd.notna(row['Genres']) else "."
            })
            ids.append(str(row['Wikipedia ID']))
        self.collection.add(
        documents=documents,
        metadatas=metadata,
        ids =ids  
        )
    def _Chroma_General_info_(self,General_Collection:dict):#### add to the General Schema
        ids=[]
        metadatas=[]
        contents=[]
        for i, category in enumerate(General_Collection):
            for j,policy in enumerate(General_Collection[category]):
                ids.append(f"{str(i)}_{str(j)}")
                metadatas.append({"category":category,"pilicy":policy})
                contents.append(General_Collection[category][policy])
            
        self.General_collection.add(
        documents=contents,
        metadatas=metadatas,
        ids =ids  
        )
        return f"""{len(ids)} has been added to the [general] Collection """

    def _Chroma_Search_(self,Query:str,User_ID:str,Task="recommend"):
        
        
        #User_ID="Babak_EA"
        self.user["ID"]=User_ID
        
        report=self.collection.query(
            #query_texts=["farm animal","titel: punk"],
            query_texts=[Query],
            n_results=20,
            #where={"title": "punk "},
            #where_document={"$contains":"search_string"}
            )
        Report={}
        for i,j in enumerate(report["distances"][0]):
            if j<= 1.1:
                #report["distances"][0]
                Report[i]={
                    "title":report["metadatas"][0][i]["title"],
                    "author":report["metadatas"][0][i]["author"],
                    "genres":report["metadatas"][0][i]["genres"],
                    "documents":report["documents"][0][i],
                }
        if len(report)>=1:
        
            User_info={"ID":self.user["ID"],"Query":Query,"Data":Report}  
            
            if Task=="recommend":

                Open_AI_response=self._Open_AI_Recommender_(query=User_info,Task="recommend")
            else:
                Open_AI_response=self._Open_AI_summarizer_(query=User_info,Task="recommend")
                
            self.Search_results=Report


            return Open_AI_response,Report
        else:
            return "file not found","file not found"
        
    def _Chroma_General_Search_(self,Query:str,User_ID:str,Task="recommend"):
        #User_ID="Babak_EA"
        self.user["ID"]=User_ID
        
        report=self.General_collection.query(
            #query_texts=["farm animal","titel: punk"],
            query_texts=[Query],
            n_results=20,
            #where={"title": "punk "},
            #where_document={"$contains":"search_string"}
            )
        Report={}
        
        temp=""
        for i in Report:

            temp+=f" {Report[i]['policy']}, {Report[i]['documents']}  "
        temp 
        ####metadatas'[category],metadatas'[pilicy],documents,ids
        for i,j in enumerate(report["distances"][0]):
            if j<= 1.1:
                #report["distances"][0]
                Report[i]={
                    "category":report["metadatas"][0][i]["category"],
                    "policy":report["metadatas"][0][i]["pilicy"],
                    "documents":report["documents"][0][i],
                }
                temp+=f'''{report["metadatas"][0][i]["pilicy"]}, {report["documents"][0][i]} ''' 
        if len(temp)>=1:
        
            User_info={"ID":self.user["ID"],"Query":Query,"Data":temp}  
            
            if Task=="recommend":

                Open_AI_response=self._Open_AI_General_Policy_(query=User_info,Task="recommend")
            else:
                Open_AI_response=self._Open_AI_summarizer_(query=User_info,Task="recommend")
                
            self.Search_results=Report


            return Open_AI_response,Report
        else:
            return "file not found","file not found"
        
    def _Register_Chroma_(self):
        self._csv_reader_()
        self._Chroma_Generator_()
    def _collect_user_(self,collect:{}):
        pass
  
    def _Open_AI_Recommender_(self,query:dict,Task="recommend"):
        user_ID=query["ID"]
        user_query=query["Query"]
        documents_content=query["Data"]

        USER_Prompt = f"""
                    recommender:
                    user asked for {user_query}\.
                    if the user intentenion to search about the book  do :\
                        the given documants includes some selected book that has title, genrs, authos and book summary. 
                        recomman up to 3 books from beginners to expert 
                        the report woylbe like : 
                        Certainly! For beginners, I recommend '[Book name] ' by [Author].
                        [ its a very short summary of the book by one sentence]  
                        If you're looking for something more advanced,
                        '[second book title]' by [authors of the second book] and its about [a very short summary of the book by one sentence] is excellent. 

                        Would you like more information on any of these books?"
                    else if wants mor information about the book like :teme more abouy[book title] do :\
                        write a summary up to 400 tokanes and return.
                    esle if user asked for the store policies, shipping, and returns, return the the detected intend like : returns
                    .
                    books informations :
                    {documents_content}
                    """

        client = OpenAI(
        # This is the default and can be omitted
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            #stop=["\n"],
                            )
        
        Contecnt=chat_completion.choices[0].message.content
        
        Total_tokens=chat_completion.usage.__dict__["total_tokens"]
        #Total_cost=Total_tokens*0.000003
        self.Total_tokens+=Total_tokens
        self.Total_cost+=Total_tokens*0.00209
        
        report=Contecnt.replace("'", '"').replace("\n","")
        try:
            AI_Recommender=json.loads(report)
        except:
            AI_Recommender=report
    
        return AI_Recommender

               
 
    def _Open_AI_Intention_recognation_(self,query:dict):
        user_ID=query["ID"]
        #user_query =query["Query"]
        documents_content=query["Data"]

        USER_Prompt = f"""
            Intention Detection:

            User asked about {documents_content}.

            If asked about the policy, return policy details, or general information:
                "Intention": "General"

            If user inquires about a book, author, or genre without specifying a particular book:
                "Intention": "Book_Search"
                "Content": "What user is looking for in the book contents."
                "Title": "The title of the book user is looking for."
                "Genre": "The genre user is interested in."

            If user requests more information about a specific book:
                "Intention": "Book_Summary"
                "Title": "The title of the selected book."

            If user confirms or the intention is confirmation:
                "Intention": "Confirmed"

            If user declines or wishes to cancel the shopping:
                "Intention": "Cancel"

            Please return answers in a Python dictionary format.
        """

        client = OpenAI(
        # This is the default and can be omitted
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            #stop=["\n"],
                            )
        
        Contecnt=chat_completion.choices[0].message.content
        
        Total_tokens=chat_completion.usage.__dict__["total_tokens"]
        #Total_cost=Total_tokens*0.000003
        self.Total_tokens+=Total_tokens
        self.Total_cost+=Total_tokens*0.00209
        
        report=Contecnt.replace("'", '"').replace("\n","")
        try:
            Intention_resulst=json.loads(report)
        except:
            Intention_resulst=report
    
        return Intention_resulst
        

    
    #######################################################################################################3
    ########################NER ##########################################################################
    
    def _Open_AI_NER_(self,query:dict):
        user_ID=query["ID"]
        #user_query=query["Query"]
        documents_content=query["Data"]
        ###topic: means the most related book name , domain, author name or book genre

        USER_Prompt = f"""
        Detect Entities and Intention:

        NER Function: Extract book titles, author names, and genres from user input.
        Extract intention: [searching about book, information about book, purchasing, cancelling] from user input.

        Given user input {documents_content}, use the following pattern:

        User: "Can you recommend a book on [topic]?"
        Entities:  topic
        Intention: searching

        User: "Tell me more about [book title]."
        Entities: book title
        Intention: information

        User: "How can I purchase [book title]?"
        Entities:book title
        Intention: purchasing

        User: "How can I purchase"
        Entities: topic
        Intention:purchasing

        User: "How to buy then"
        Entities: None
        Intention: purchasing

        User: "Cancel my order."
        Entities: None
        Intention:cancelling

        User: "What is the return policy?"
        Entities: None
        Intention:cancelling

        User: "Who is the author of [book title]?"
        Entities: book title
        Intention: information

        User: "Show me books in the [genre]."
        Entities: genre
        Intention: searching

        User: "What is [book title] about?"
        Entities: book title
        Intention: information

        User: "Add [book title] to my cart."
        Entities:  book title
        Intention: purchasing

        User: "That would be great, thanks!"
        Entities: None
        Intention:confirmed
        
        User: " thanks! I want it"
        Entities: None
        Intention:confirmed
        
        User: " Lets do that "
        Entities: None
        Intention:confirmed

        Please return answers in a Python dictionary format.
        """

        client = OpenAI(
        # This is the default and can be omitted
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            )

        report=chat_completion.choices[0].message.content.replace("'", '"').replace("\n","")
        try:
            NER_resulst=json.loads(report)
        except:
            NER_resulst=report
    
        return NER_resulst
    
    ##########################################################################################33
    ################################## Summarizer#############################################
    
    def _Open_AI_summarizer_(self,query:dict,Task="recommend"):
        user_ID=query["ID"]
        user_query=query["Query"]
        documents_content=query["Data"]

        USER_Prompt = f"""
            Summarizer Function:

            User asked for "{user_query}".
            If the user wants more information about the book like "Tell me more about [book title]", write a summary up to 400 tokens and return it.
            
            NER Function:
            Extract the book title from the user query.
            
            
            Summarized: [summarized content].
            BookTitle: [extracted title].
            
            Return Report answeres in a text format

            The content to be summarized:

            {documents_content}
            """

        client = OpenAI(
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            )
        
        Contecnt=chat_completion.choices[0].message.content
        
        Total_tokens=chat_completion.usage.__dict__["total_tokens"]
        #Total_cost=Total_tokens*0.000003
        self.Total_tokens+=Total_tokens
        self.Total_cost+=Total_tokens*0.00209
        
        temp=Contecnt.replace("'", '"').replace("\n","")
        try:
            summarizer_response=json.loads(temp)
        except:
            summarizer_response=temp
        return summarizer_response
    
    
    def _Open_AI_Reommend_Summeriser_(self,documents_content):

        USER_Prompt = f"""
            Summarizer Function:
            given document, tell the storuy in only one sntence not longer that 150 charackters
            use this content:
            {documents_content}
            """
        client = OpenAI(
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            )
        
        Contecnt=chat_completion.choices[0].message.content
        temp=Contecnt.replace("'", '"').replace("\n","")
        try:
            summarizer_response=json.loads(temp)
        except:
            summarizer_response=temp
        return summarizer_response    

    
    ######################################################################################
    ####################### Policy and General Search #####################################
    def _Open_AI_General_Policy_(self,query:dict,Task="recommend"):
        user_ID=query["ID"]
        user_query=query["Query"]
        documents_content=query["Data"]

        USER_Prompt = f"""
        Summaraizer Function:

        User Question: "{user_query}"
        Given the provided documents, search for the appropriate answers to the user's question.

        the generated answer
        
        eturn the answer in text format.
        The provided documents for search:

        {documents_content}
        """

        client = OpenAI(
        api_key=self.api_key,)
        chat_completion = client.chat.completions.create(
                            messages=[
                                {
                                    "role": "user",
                                    "content": USER_Prompt,
                                }
                            ],
                            model="gpt-3.5-turbo",
                            max_tokens=2000,
                            )
        
        Contecnt=chat_completion.choices[0].message.content
        
        Total_tokens=chat_completion.usage.__dict__["total_tokens"]
        #Total_cost=Total_tokens*0.000003
        self.Total_tokens+=Total_tokens
        self.Total_cost+=Total_tokens*0.00209
        
        temp=Contecnt.replace("'", '"').replace("\n","")
        try:
            summarizer_response=json.loads(temp)
        except:
            summarizer_response=temp
        return summarizer_response
    
    
    
    #######################################################################################
    def _Start_Agent_(self,query:str,User_ID:str):
        if User_ID in self.user:
            pass
        else:
            self.user[User_ID]={"Query":[],"Response":[],"Content":[]}
        self.user[User_ID]["Query"].append(query)

        Search_Query={"ID":User_ID,"Data":query}
        temp=json.loads(self._Open_AI_Intention_recognation_(query=Search_Query).choices[0].message.content.replace("'", '"'))
        if temp["intention"] == "Book_Search":
            Ansswer,Content=self._Chroma_Search_(Query=query,User_ID=User_ID)
            self.user[User_ID]["Response"].append(Ansswer)
            self.user[User_ID]["Content"].append(Content)
            
        elif temp["intention"] == "Book_Search":
            Ansswer,Content=self._Chroma_Search_(Query=query,User_ID=User_ID)
            self.user[User_ID]["Response"].append(Ansswer)
            self.user[User_ID]["Content"].append(Content)
            
        elif temp["intention"] == "General":
            pass
        
        
        from fuzzywuzzy import fuzz

    def select_closest_titles_fuzzy(self,extracted_title:str, title_list:list):
        title_similarities = [(title, fuzz.ratio(extracted_title.lower(), title.lower())) for title in title_list]
        sorted_titles = sorted(title_similarities, key=lambda x: x[1], reverse=True)
        #closest_titles = sorted_titles[:top_n]
        closest_titles=[x[0] for x in sorted_titles if x[1]>=80]

        return closest_titles
        
    ################################################################################################################3
    ######################################### Dessision ############################################################
    
    def _Desission__(self,Results:dict,User_Query:str,User_ID="Babak_EA"):
        if User_ID in self.user_cart:
            pass
        else:
            self.user_cart[User_ID]={"searching":{"Raw_data":[],"LLM_response":[]},
                                "Summarized":{"LLM_response":[],"Book_Titles":[]},
                                "Policy":[],
                               "entities":[],
                               "books":[],
                               "genres":[],
                               "Authors":[],
                                "Bought":False }
        # List of keywords indicating the end of the conversation
        end_keywords = ["thanks", "thank you", "bye", "goodbye", "see you", "thats all "]

        # User input string
        user_input = str(User_Query)

        # Check if any keyword is present in the user input
        conversation_ended = any(keyword in user_input.lower() for keyword in end_keywords)

        if conversation_ended:
            Results["intention"]="confirmed"

            

        if str(Results["intention"])=="searching":
#             if Results['entities']:
#             #Query=f"{User_Query} + includes {Results['entities']}"
#                 Query={Results['entities']}
#             else:
#                 Query=User_Query
            
            Query=f"{User_Query}"
            Open_AI_response,Report=self._Chroma_Search_(Query=Query,User_ID=User_ID)####351
            self.user_cart[User_ID]["searching"]["Raw_data"].append(Report)
            self.user_cart[User_ID]["searching"]["LLM_response"].append(Open_AI_response)

        elif str(Results["intention"])=="information":
            Book_Content={}
            
            #print(f'''printing the book  {Results["entities"]}''')

            try:
                detected_book=Results["entities"]["book_title"]
            except:
                detected_book=Results["entities"]
            self.TEST_BOOK=detected_book
                

            Summarised={"Summarized data":"","Book Title":""}

            if type(detected_book)==list:
                detected_book=detected_book[0]

                
            ########################### updated #######################
            try:
            
                for key in self.user_cart[User_ID]["searching"]["Raw_data"][0]:
                    tem=self.user_cart[User_ID]["searching"]["Raw_data"][0][key]
                    try:

                        if fuzz.ratio(detected_book, tem["title"].lower())>70:
                            Book_Content[tem["title"]] =tem["documents"]
                        else:
                            pass
                    except:
                        Book_Content={}
                    
            except:
                 Book_Content={}
#                 ######## if the user cart is empty , chage the intentiopn
#                 Results["intention"]=="searching"
#                 self._Desission__(Results,User_Query,User_ID)
                                
                

#             for key in self.user_cart[User_ID]["searching"]["Raw_data"][0]:
#                 tem=self.user_cart[User_ID]["searching"]["Raw_data"][0][key]

#                 if fuzz.ratio(detected_book.lower(), tem["title"].lower())>85:
#                     Book_Content[tem["title"]] =tem["documents"]
                    
            
            if len(Book_Content)>0:

                Query={"ID":User_ID,"Query":User_Query,"Data":Book_Content}
                Summarised=self._Open_AI_summarizer_(query=Query,Task="recommend")
            else:
                #Query=query,User_ID=User_ID
                Summarised,Report=self._Chroma_Search_(Query=User_Query,User_ID=User_ID,Task="summarized")
            try:    
                Summarised=Summarised.split("Summarized:")[1].replace('"',"").strip()
            except:
                pass
            try:
                Summarised,Book=Summarised.split("BookTitle:")
                self.user_cart[User_ID]["Summarized"]["LLM_response"].append(Summarised)
                self.user_cart[User_ID]["Summarized"]["Book_Titles"].append(Book.strip())
            except:
                self.user_cart[User_ID]["Summarized"]["LLM_response"].append(Summarised)
                self.user_cart[User_ID]["Summarized"]["Book_Titles"].append(Summarised)


        elif str(Results["intention"]) in ["cancelling","purchasing"]:
            Query={"ID":User_ID,"Query":User_Query}
            Query=User_Query
            Policy,Report=self._Chroma_General_Search_(Query=Query,User_ID=User_ID,Task="recommend")


            self.user_cart[User_ID]["Policy"]=[Results["intention"],Policy]
            if Results["intention"]=="cancelling":
                #### recommendation:
                return f"Underestat {User_ID} I will clear your cart. Hope to see you soon"

            else:

                return f"""
                You're welcome {User_ID}! I've added  {self.user_cart[User_ID]["Summarized"]["Book_Titles"]} 
                to your cart.
                You can proceed to checkout whenever you're ready.Is there anything else I can help you with?"""

        elif str(Results["intention"]) in ["confirmed","ok","thanks"] :

            ####Recommendation
            return "Is was a pleasure working with you. Is there anything else I can help you with?"
        else :
            return "Intention not found"
       

            ####Recommendation
            return "Is was a pleasure working with you. Is there anything else I can help you with?"
    def _Preper_for_Cart_(self,User_ID:str,User_Query:str):
        self.Dict_Hist={"User_ID":User_ID,
                        "book_title":"",
                        "genre":"",
                        "book_author":"",
                        "search":User_Query
                       }
        for i in self.user_cart[User_ID]["searching"]["Raw_data"][0]:
        #     print(i)
            if self.user_cart[User_ID]["searching"]["Raw_data"][0][i]["title"]==self.user_cart[User_ID]["Summarized"]["Book_Titles"][0]:
                Temp=self.user_cart[User_ID]["searching"]["Raw_data"][0][i]
#                 Dict_Book.append([Temp["title"],
#                                  Temp["author"],
#                                 Temp["genres"]])
                self.Dict_Hist["book_title"]=Temp["title"]
                self.Dict_Hist["book_author"]=Temp["author"]
                self.Dict_Hist["genre"]=Temp["genres"]
                
                                                                                               
    ##################################################################################################                                                                                        
    ################################ Basic Recommender ###############################################3
    def Recommender_Grnre(self,User_ID:str):
        Log_search=self.Log_Database.get_search_hist(User_ID)
        Genre=[] #### search based on the Youser Genres
        if len(Log_search)>0:
            if len(Log_search[0])==5:
                Genre.append([x[4] for x in Log_search])

            else:
                Genre=[Log_search[4]]
        if len(Genre)>0:
            Genre=self._flatten_(Genre)
            Genre=list(set(Genre))

        Recommended=self._Recommend_Genre_(Genre,num_records=5)
        
        ##### Summarizing the Books abstracts:
        for title, documents in Recommended.items():
            #print(f"{title}: {documents}\n") 
            Recommended[title]=self._Open_AI_Reommend_Summeriser_(documents)
        
        
        
        return Recommended        

    def _select_random_records_(self,Report:dict, num_records=4):
        random_records = random.sample(Report.items(), num_records)

        return dict(random_records)

    def _reorganize_report_(self,Report:dict):
        titles_documents_dict = {}

        for metadata, documents in zip(Report['metadatas'][0], Report['documents'][0]):
            title = metadata['title']
            titles_documents_dict[title] = documents

        return titles_documents_dict

    def _Recommend_Genre_(self,Genre:list,num_records=5):
        """
        list of the genres, 
        """
        Recomand = self.collection.query(
        query_texts=[""],
        n_results=10000,
        where={"genres": {"$in": Genre}},
        include=[ "documents","metadatas" ])

        Recomand=self._reorganize_report_(Recomand)
        Recomand=self._select_random_records_(Recomand, num_records)
        return Recomand

    def _flatten_(self,lst):
        flattened_list = []
        for item in lst:
            if isinstance(item, list):
                flattened_list.extend(self._flatten_(item))
            else:
                flattened_list.append(item)
        return flattened_list


        
        

    ##################################################################################################
    ################################## _CalL_ ######################################################
    def _Call_(self,Question:str,User_ID:str):
        #Question=Question
        query={"ID":User_ID,"Data":Question}

        Results=self._Open_AI_NER_(query=query)
        #print(Results)

        #Results={key.lower():Results[key] for key in Results}
        if type(Results)!=dict:
            try:
                Results=json.loads(Results)
            except:
                pass
        try:
            Results = {key.lower(): value.lower() if isinstance(value, str) else value for key, value in Results.items()}
        except:
            Results=Results

        chatbot_message=self._Desission__(Results=Results,User_Query=Question,User_ID="Babak_EA")
        if Results['intention']=="searching":
            
            return (str(self.user_cart[User_ID]["searching"]["LLM_response"]))
        
        elif Results['intention']=="information":
            #self._Preper_for_Cart_(self,User_ID:str,User_Query:str)
            #print("***** first !!!")
            
            
            
            self._Preper_for_Cart_(User_ID=User_ID,User_Query=Question)
            #print("Save to Dataset")
            self.Log_Database.insert_search_hist(self.Dict_Hist["User_ID"], 
                                                 self.Dict_Hist["book_title"],
                                                 Question,self.Dict_Hist["genre"])

            return (str(self.user_cart[User_ID]["Summarized"]))

        elif Results['intention'] in ["cancelling","purchasing"]:
            if Results['intention']=="purchasing":
                self.user_cart[User_ID]["Bought"]=True
            elif Results['intention']=="cancelling":
                self.user_cart[User_ID]["Bought"]=False
                self.Log_Database.insert_search_book(self.Dict_Hist["User_ID"],self.Dict_Hist["book_title"],
                                   self.Dict_Hist["book_author"],
                                   self.Dict_Hist["genre"],bought=False )
            return (self.user_cart[User_ID]["Policy"], str(chatbot_message))
        
        elif Results['intention']in ["confirmed","ok","thanks"]:
            
            self.Log_Database.insert_search_book(User_ID,self.Dict_Hist["book_title"],
                        self.Dict_Hist["book_author"],
                        self.Dict_Hist["genre"],bought=True )
            self.Log_Database.insert_user_bonus(User_ID,10 )
            
            ##### Recommender Call
            Recommends_books = self.Recommender_Grnre(User_ID)

            Report = {
                "Chatbot": f"""It was a pleasure working with you '{User_ID}'. 
                Is there anything else I can help you with?
                By the way, as you liked the '{self.Dict_Hist["book_title"]}', 
                you may also want to take a look at the following masterpieces:"""
            }

            for key, value in Recommends_books.items():
                Report[key] = value
                
            

            return Report
        
#         _Preper_for_Cart_(self,User_ID:str,User_Query:str,User_Bout:bool)

########################################################################################################



########################################################################################################


