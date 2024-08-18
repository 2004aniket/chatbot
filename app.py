# kwRoBdXp_eNk9xAYzm26dMGM1PRVspFzETd68PaCm
import os
import streamlit as st
from ultralytics import YOLO
import numpy as np
from langchain_mistralai import ChatMistralAI
# from langchain import LLMChain,PromptTemplate
os.environ["MISTRAL_API_KEY"] = 'kgYDKvLmZ6tPlU2CyFFY2S642npGBHEM'
model = ChatMistralAI(model="mistral-large-latest")
# template = ''' 
#           ANSWER: 
#   '''

# prompttemp = PromptTemplate (template = template, input_variables = ['scenario'])
choice=""
model = YOLO('yolov8n.pt')
# def getimagedetect():
    
def getresponse(prompt):
    if prompt:
        # req=LLMChain(llm=model,prompt=prompttemp)
        # response=req.invoke(prompt)
        response=model.predict(prompt)
        # st.write(response)
        return response

st.title("chatbot")

with st.sidebar:
    choice=st.radio('Pick one:', ['text','image'])

    
if "messages" not in st.session_state:
    st.session_state.messages = []


#    st.line_chart(np.random.randn(30, 3))
# response=getresponse(prompt=prompt)

# Display a chat input widget.

#file=st.file_uploader("upload your image")


# if file:
#     st.download_button('Download Jpg', file, 'detect.jpg')
#     detect()
    
if choice=="text":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
             st.markdown(message["content"])

    if prompt:=st.chat_input("whats up"):
        with st.chat_message("user"):
            st.markdown(prompt)

    st.session_state.messages.append({"role":"user","content":prompt})
    response=getresponse(prompt=prompt)
    with st.chat_message('ai'):
        st.markdown(response)
    st.session_state.messages.append({"role":"ai","content":response})
else:
    file=st.file_uploader("upload your image")
    if file:
        col1,col2=st.columns(2)
        st.image(file)
        results=model("detect.jpg")
        annotated_frame=results[0].plot()
        org_frame=col1.empty()
        ann_frame=col2.empty()
        ann_frame.image(annotated_frame,channels="BGR")
    # os.path.exists("detect.jpg")
    # col2 = st.columns(1)
    # results = model("detect.jpg")
    # annotated_frame = results[0].plot()  # Add annotations on frame
    # # org_frame = col1.empty()
    # ann_frame = col2.empty()
    # # org_frame.image(file, channels="BGR")
    # ann_frame.image(annotated_frame, channels="BGR")
    
