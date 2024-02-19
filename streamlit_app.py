import streamlit as st
import requests

url = "http://127.0.0.1:9900/"

# =============================================================================
# params = {
#     "User_ID": "Babak_EA",
#     "Question": "Can you recommend a book on farm animal"
# }
# =============================================================================

headers = {
    "accept": "application/json"
}

# =============================================================================
# 
# 
# 
# 
# # Define functions to interact with your API
# def search_user(userID):
#     # Make an API call to search for user by userID
#     # Replace the URL with your actual API endpoint
#     response = requests.get(f"https://your-api-url.com/search_user?userID={userID}")
#     if response.status_code == 200:
#         return response.json()
#     else:
#         return None
# 
# def insert_user(user_info):
#     # Make an API call to insert user information
#     # Replace the URL with your actual API endpoint
#     response = requests.post("https://your-api-url.com/insert_user", json=user_info)
#     return response.ok
# 
# def ask_question(question):
#     # Make an API call to get a response to the question
#     # Replace the URL with your actual API endpoint
#     response = requests.get(f"https://your-api-url.com/ask_question?question={question}")
#     if response.status_code == 200:
#         return response.json()["response"]
#     else:
#         return "Failed to get response from API"
# 
# # Streamlit UI
# st.title("Chatbot User Interface")
# 
# # Page for searching by userID
# st.subheader("Search User by UserID")
# userID = st.text_input("Enter UserID:")
# if st.button("Search"):
#     if userID:
#         user_info = search_user(userID)
#         if user_info:
#             st.write("User Found:")
#             st.write(user_info)
#         else:
#             st.write("User not found.")
# 
# # Page for inserting user information
# st.subheader("Insert User Information")
# name = st.text_input("Name:")
# email = st.text_input("Email:")
# insert_button = st.button("Insert User")
# if insert_button and name and email:
#     user_info = {"name": name, "email": email}
#     if insert_user(user_info):
#         st.write("User information inserted successfully.")
#     else:
#         st.write("Failed to insert user information.")
# 
# # Page for asking questions
# st.subheader("Ask Question")
# userID_for_question = st.text_input("Enter UserID:")
# question = st.text_input("Enter your question:")
# if st.button("Ask") and question and userID_for_question:
#     response = ask_question(userID_for_question, question)
#     st.write("Response:")
#     st.write(response)
# =============================================================================









import streamlit as st
import requests
URL = "http://127.0.0.1:9900/"
headers = {"accept": "application/json"}
# Define functions to interact with your API
import json
def call_chat_api(user_id, question):
    url = "http://127.0.0.1:9900/Chat/"
    params = {
        "User_ID": user_id,
        "Question": question
    }
    headers = {
        "accept": "application/json"
    }
    try:
        response = requests.post(url, params=params, headers=headers)
        response.raise_for_status()  # Raise an exception for 4xx or 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"Request failed: {e}"}
    except json.decoder.JSONDecodeError:
        return {"error": "Failed to decode JSON response"}






def search_user(userID):
    # Make an API call to search for user by userID
    url = f"{URL}user_serach"
    params = {"User_ID": userID}
    response = requests.post(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def insert_user(user_id, user_email, user_phone):
    # Make an API call to insert user information
    # Replace the URL with your actual API endpoint
    params = {
        "User_ID": user_id,
        "email_address":user_email,
        "user_phone":str(user_phone)
     }
    url = f"{URL}user_registry/"
    response = requests.post(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()
    else:
        return None
    
    
    
def ask_question(userID, question):
    # Make an API call to get a response to the question
    # Replace the URL with your actual API endpoint
    url = f"{URL}Chat"
    params = {"User_ID": userID, "question": question}
    headers = {"accept": "application/json"}
    response = requests.post(url, params=params, headers=headers)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return "Failed to get response from API"




# Streamlit UI
st.title("Chatbot User Interface")

# Page for searching by userID
st.subheader("Search User by UserID")
userID = st.text_input("Enter UserID:")
if st.button("Search"):
    if userID:
        user_info = search_user(userID)
        if user_info:
            st.write("User Found:")
            st.write(user_info)
        else:
            st.write("User not found.")

# Page for inserting user information
# Page for inserting user information
st.subheader("Insert User Information")
user_id = st.text_input("User ID:")
user_email = st.text_input("Email Address:")
user_phone = st.text_input("Phone Number:")
insert_button = st.button("Insert User")
if insert_button and user_id and user_email and user_phone:
    if insert_user(user_id, user_email, user_phone):
        st.write("User information inserted successfully.")
    else:
        st.write("Failed to insert user information.")

# Page for asking questions
st.subheader("Ask Questions")
user_question = st.text_area("Enter Your Question:", height=100)
ask_button = st.button("Ask")
if ask_button and user_question:
    # Call your function to process the question here
    response = ask_question(userID_for_question, question)
    st.write("Response:")
    st.write(response)

















