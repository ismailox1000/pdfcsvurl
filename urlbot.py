import os
import openai
import streamlit as st


def read_url(url_link, questions):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"i will provide you a url and you will give me deep answers based on my question\nurl:{url_link}  \nquestion:{questions}\nanswer:\n",
        temperature=0.0,
        max_tokens=2500,
        top_p=1,
        frequency_penalty=0.25,
        presence_penalty=0.2
    )
    answer_list = response['choices'][0].text.split("\n")
    return answer_list


#tools = st.sidebar.selectbox("Select a tool", ("Blog Writer", "Sentence Paraphraser",'Summarize Text'))
st.title("Blog Post Generator")

# Get user topic input
url_link = st.text_input("Enter your url:")
question=st.text_input('what is your question')
if url_link and question:
    if st.button('generate answer'):
        with st.spinner("Generating answer..."):
            st.write("Here is your answer:")
            answer_list = read_url(url_link=url_link, questions=question)
            
            st.write(answer_list)
            

