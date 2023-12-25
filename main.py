import streamlit as st
from ai_bot import get_qa_chain, create_vectorDb

st.title("Ask me anything")
btn = st.button("Create Knowledgebase")
if btn:
    create_vectorDb()
    st.success("Knowledge base creation completed successfully!") 


question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    st.header("Answer")
    st.write(response["result"])
    st.header('Source') 
    st.write(response['source_documents'])  


