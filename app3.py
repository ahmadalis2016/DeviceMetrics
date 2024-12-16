import streamlit as st 
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Configure OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=openai_api_key)

# Incident database (in-memory for simplicity)
incident_db = [
    {"Incident Number": "INC7900003", "Short Description": "Zoom video stopping"},
    {"Incident Number": "INC7900021", "Short Description": "Unable to launch Tableau"}
]

# Debounce logic
last_call_time = 0
debounce_interval = 2  # Seconds

def debounce_request():
    global last_call_time
    current_time = time.time()
    if current_time - last_call_time < debounce_interval:
        return False
    last_call_time = current_time
    return True

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_text(text)

# Function to load JSON data and process it
def process_json_data(file_path):
    df = pd.read_json(file_path)
    raw_text = df.to_string(index=False)
    text_chunks = get_text_chunks(raw_text)
    return text_chunks, df

# Function to create the QA chain
def get_qa_chain(retriever):
    llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0.3)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa_chain

# Function to handle device-related questions dynamically for specific fields
def handle_device_question(device_id, df, field_name=None):
    # Filter the dataframe for the specific device
    device_data = df[df["Device_ID"] == device_id]
    if device_data.empty:
        return f"No data found for device {device_id}. Please check the device ID."
    else:
        if field_name:
            # Check if the requested field exists in the data
            if field_name in df.columns:
                field_data = device_data[field_name]
                return field_data.to_string(index=False)
            else:
                return f"The field '{field_name}' is not found in the data. Available fields are: {', '.join(df.columns)}"
        else:
            # Return all data for the device if no specific field is requested
            return device_data.to_string(index=False)



# Function to handle user questions and provide answers
def handle_user_question(user_question, retriever, df):
    device_id = extract_device_id(user_question)
    if device_id:
        if "incidents" in user_question.lower():
            return handle_device_question(device_id, df, "Incidents")
        elif "application crashes" in user_question.lower():
            return handle_device_question(device_id, df, "Application_Crashes")
        else:
            # If no specific field is mentioned, return all data for the device
            return handle_device_question(device_id, df)
    
    # If no device-specific question, use the general QA chain
    qa_chain = get_qa_chain(retriever)
    response = qa_chain.run(user_question)

    if "recommendation" in user_question.lower() or "suggestion" in user_question.lower():
        recommendations = generate_recommendations(user_question, df)
        response += f"\n\n**Recommendations:**\n{recommendations}" if recommendations else "\n\nNo specific recommendations available."

    return response


# Utility function to extract a device ID from user input
def extract_device_id(question):
    words = question.split()
    for word in words:
        if word.startswith("L") and word[1:].isdigit():
            return word
    return None

# Function to generate context-aware recommendations
def generate_recommendations(user_question, df):
    recommendations = []

    if "RAM" in user_question or "ram" in user_question.lower():
        if df["Application_Crashes"].str.contains("RAM", case=False, na=False).any():
            recommendations.append("For devices with RAM issues: Close high RAM usage applications or restart the device.")

    if "Zoom" in user_question or "zoom" in user_question.lower():
        if df["Application_Crashes"].str.contains("Zoom", case=False, na=False).any():
            recommendations.append("For devices with Zoom issues: Check for appropriate plugins or contact the Zoom administrator.")

    if "Microsoft" in user_question.lower() or "outlook" in user_question.lower():
        if df["Application_Crashes"].str.contains("Microsoft", case=False, na=False).any():
            recommendations.append("For Microsoft-related issues: Refer to the Microsoft Online Knowledge Base for solutions.")

    if "stability index" in user_question.lower():
        threshold = 8.0
        high_stability = df[df["Stability_Index"] > threshold]
        if not high_stability.empty:
            recommendations.append(f"There are {len(high_stability)} devices with a stability index over {threshold}. Consider reviewing their configurations or performance trends.")
        else:
            recommendations.append("No devices found with a stability index above the specified threshold.")

    if "missing data" in user_question.lower():
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            recommendations.append(f"The dataset contains {missing_count} missing values. It's recommended to handle them via imputation or removal.")

    return "\n".join(recommendations)

# Function to create an incident
def create_incident(description):
    # Generate next incident number (based on the last incident number in the list)
    if len(incident_db) == 0:
        next_number = 1  # Start from INC7900001 if the incident_db is empty
    else:
        last_incident_number = incident_db[-1]["Incident Number"]
        next_number = int(last_incident_number[3:]) + 1  # Increment the last number

    new_incident_number = f"INC{next_number:07d}"  # Format the number with leading zeros
    new_incident = {"Incident Number": new_incident_number, "Short Description": description}
    
    incident_db.append(new_incident)
    return new_incident

# Function to load JSON data and process it
def process_json_data(file_path):
    try:
        df = pd.read_json(file_path, lines=True)  
        raw_text = df.to_string(index=False)
        text_chunks = get_text_chunks(raw_text)
        return text_chunks, df
    except ValueError as e:
        if "Trailing data" in str(e):
            raise ValueError("The JSON file contains trailing data or invalid formatting. Please check the file.") from e
        else:
            raise


# Streamlit Interface
def main():
    st.set_page_config(page_title="Continuous Q&A Chat with Device Health Data", layout="wide")
    st.title("Continuous Q&A Chat with Device Health Data")

    # Initialize session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]

    # Read and process JSON data
    file_path = "Data/Device_Health_Data.json"
    if os.path.exists(file_path):
        text_chunks, df = process_json_data(file_path)
        faiss_index = FAISS.from_texts(text_chunks, embeddings)

        st.success("Data loaded successfully.")

        # Display or clear chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])

        def clear_chat_history():
            st.session_state.messages = [{"role": "assistant", "content": "How may I assist you today?"}]
        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

        # User-provided question
        user_question = st.chat_input("Ask a question about the data:")
        if user_question and debounce_request():
            st.session_state.messages.append({"role": "user", "content": user_question})
            with st.chat_message("user"):
                st.write(user_question)

            # Generate a new response if the last message is not from the assistant
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        if "create an incident" in user_question.lower():
                            description = user_question.replace("create an incident for", "").strip()
                            incident = create_incident(description)
                            response = f"Incident created successfully!\n\nIncident Number: {incident['Incident Number']}\nShort Description: {incident['Short Description']}"
                        else:
                            response = handle_user_question(user_question, faiss_index.as_retriever(), df)
                        st.write(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})

    else:
        st.error(f"File not found at path: {file_path}")


if __name__ == "__main__":
    main()
