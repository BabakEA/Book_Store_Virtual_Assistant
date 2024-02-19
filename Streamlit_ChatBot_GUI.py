import streamlit as st
import requests

# Function to call the FastAPI endpoint for Chat
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

# Function to call the FastAPI endpoint for user search
def call_user_search_api(user_id):
    url = "http://127.0.0.1:9900/user_serach/"
    params = {
        "User_ID": user_id
    }
    response = requests.post(url, params=params,headers=headers)
    return response.json()

# Function to call the FastAPI endpoint for user registration
def call_user_registry_api(user_id, email_address, user_phone):
    url = "http://127.0.0.1:9900/user_registry/"
    params = {
        "User_ID": user_id,
        "email_address": email_address,
        "user_phone": user_phone
    }
    response = requests.post(url, params=params,headers=headers)
    return response.json()

# Function to call the FastAPI endpoint for user info
def call_user_info_api(user_id):
    url = "http://127.0.0.1:9900/user_info/"
    params = {
        "User_ID": user_id
    }
    response = requests.post(url, params=params,headers=headers)
    return response.json()

# Function to call the FastAPI endpoint for genre recommender
def call_genre_recommender_api(user_id):
    url = "http://127.0.0.1:9900/Genres_Recommender/"
    params = {
        "User_ID": user_id
    }
    response = requests.post(url, params=params,headers=headers)
    return response.json()

# Streamlit GUI
# def main():
#     st.title("Chatbot")

#     # Sidebar navigation
#     page = st.sidebar.selectbox("Select Page", ["Chat", "User Search", "User Registry", "User Info", "Genre Recommender"])

#     if page == "Chat":
#         st.subheader("Chat")
#         user_id = st.text_input("Enter Your ID")
#         question = st.text_area("Ask your question")

#         if st.button("Ask"):
#             if user_id and question:
#                 response = call_chat_api(user_id, question)
#                 if isinstance(response, str):
#                     st.write(response)
#                 else:
#                     st.write("Bot's Response:")
#                     st.write(response.get("response", "No response received from the server."))

#     elif page == "User Search":
#         # Rest of the code remains the same

# Streamlit GUI
# Streamlit GUI
# def main():
#     st.title("Chatbot")

#     # Initialize chat history
#     chat_history = []

#     # Sidebar navigation
#     page = st.sidebar.selectbox("Select Page", ["Chat", "User Search", "User Registry", "User Info", "Genre Recommender"])

#     if page == "Chat":
#         st.subheader("Chat")
#         user_id = st.text_input("Enter Your ID")
#         question = st.text_area("Ask your question")

#         if st.button("Ask"):
#             if user_id and question:
#                 # Append user question to chat history
#                 chat_history.append(f"User ({user_id}): {question}")

#                 # Call the API and get the response
#                 response = call_chat_api(user_id, question)

#                 # Check if response is a dictionary
#                 if isinstance(response, dict):
#                     # Append API response to chat history
#                     chat_history.append("Bot: " + response.get("response", "No response received from the server."))
#                 else:
#                     # Append string response to chat history
#                     chat_history.append("Bot: " + response)

#         # Display chat history
#         st.text("\n".join(chat_history))

##############################################################################################

def main():
    st.title("Chatbot")

    # Initialize chat history
    chat_history = []

    # Sidebar navigation
    page = st.sidebar.selectbox("Select Page", ["Chat", "User Search", "User Registry", "User Info", "Genre Recommender"])

    if page == "Chat":
        st.subheader("Chat")
        user_id = st.text_input("Enter Your ID")
        question = st.text_area("Ask your question")

        if st.button("Ask"):
            if user_id and question:
                # Append user question to chat history
                chat_history.append(f"User ({user_id}): {question}")

                # Call the API and get the response
                response = call_chat_api(user_id, question)

                # Check if response is a dictionary
                if isinstance(response, dict) and "LLM_response" in response:
                    # Append each response string to chat history
                    for item in response["LLM_response"]:
                        chat_history.append("Virtual Assistant: " + item)
                else:
                    # Append string response to chat history
                    chat_history.append("virtual Assistant: " + str(response))

        # Display chat history line by line
        for message in chat_history:
            st.text_area(message)
            
    elif page == "User Search":
        # Rest of the code remains the same

        st.subheader("User Search")
        user_id = st.text_input("Enter User ID")

        if st.button("Search"):
            if user_id:
                response = call_user_search_api(user_id)
                st.write("User Info:")
                st.write(response)

    elif page == "User Registry":
        st.subheader("User Registry")
        user_id = st.text_input("Enter Your ID")
        email_address = st.text_input("Your Email Address")
        user_phone = st.text_input("Your Phone Number")

        if st.button("Register"):
            if user_id and email_address and user_phone:
                response = call_user_registry_api(user_id, email_address, user_phone)
                st.write("Registration Result:")
                st.write(response)

    elif page == "User Info":
        st.subheader("User Info")
        user_id = st.text_input("Enter Your ID")

        if st.button("Get Info"):
            if user_id:
                response = call_user_info_api(user_id)
                st.write("User Search History:")
                st.write(response)

    elif page == "Genre Recommender":
        st.subheader("Genre Recommender")
        user_id = st.text_input("Enter Your ID")

        if st.button("Recommend"):
            if user_id:
                response = call_genre_recommender_api(user_id)
                st.write("Recommended Genres:")
                st.write(response)

if __name__ == "__main__":
    main()

###### streamlit run ChatBot_GUI.py ##############