import streamlit as st
import requests

def display_response(response_text):
    if response_text == "I don't know.":
        st.warning("The model is not sure about the answer.")
    else:
        st.subheader("Response")
        st.write(response_text)

def display_extracted_text(extracted_text):
    st.subheader("Extracted Text")
    st.write(extracted_text)

def display_relevant_urls(extracted_urls):
    st.subheader("Relevant URLs")
    for url in extracted_urls:
        st.write(url)

def display_cosine_scores(cosine_scores):
    st.subheader("Cosine Similarity Scores")
    for score in cosine_scores:
        st.write(score)

def process_query(query):
    url = "http://localhost:8080/process_query/"
    response = requests.post(url, json={"query": query})
    return response

def main():
    st.set_page_config(page_title="Document Chatbot", page_icon=":robot_face:")
    st.title("Document Chatbot :robot_face:")

    query = st.text_area("Enter your query here:")
    show_extracted_text = st.checkbox("Show extracted text")

    if st.button("Get Response"):
        response = process_query(query)

        if response.status_code == 200:
            data = response.json()
            response_text = data["response"]
            extracted_text = data["extracted_documents"]
            extracted_urls = data["urls"]
            cosine_scores = data["cosine_scores"]

            display_response(response_text)

            if show_extracted_text:
                display_extracted_text(extracted_text)

            display_relevant_urls(extracted_urls)
            display_cosine_scores(cosine_scores)
        else:
            error_message = response.json().get("detail", "Unknown error")
            st.error(f"Failed to get a response from the server. Error: {error_message}")

if __name__ == "__main__":
    main()