
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import LLMChain
from langchain.schema import SystemMessage, HumanMessage
import os

# Set up Streamlit app
st.title("Text Summarization from a PDF")
st.write("Upload a PDF and get a text summarization")

# Input the Groq API key
groq_api_key = st.sidebar.text_input("Enter the Groq API Key: ", type="password")

# Drop-down to select the summarization technique
text_tech = st.sidebar.selectbox("Select a Text Summarization Technique", ["LLMChain", "Stuff Doc", "Map-Reduce", "Refine"])

# Check if the Groq API key is provided
if groq_api_key:
    # Initialize the LLM model
    llm = ChatGroq(model_name="Gemma2-9b-It", groq_api_key=groq_api_key)
    
    # Upload the PDF
    upload_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if upload_file:
        # Save the uploaded file to a temporary file
        temp_file_path = os.path.join(os.getcwd(), "temp.pdf")
        with open(temp_file_path, "wb") as file:
            file.write(upload_file.getvalue())

        # Load the PDF with PyPDFLoader
        loader = PyPDFLoader(temp_file_path)
        document = loader.load_and_split()

        page_content = ''
        doc_content = ''

        # If document is empty, stop processing
        if not document:
            st.warning("No content extracted from the PDF.")

        else:
            # Proceed with the selected summarization technique
            if text_tech == "Stuff Doc":
                # doc_len = len(document)
                # document = (document[0:(doc_len)-1].page_content[:1000])

                text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=30, length_function=len)
                chunks = text_splitter.split_documents(document)

                # Summarization using the Stuff technique
                template = """
                    Write a concise and short summary of the following speech:
                    Speech: {text}
                """
                prompt = PromptTemplate(input_variables=['text'], template=template)
                chain = load_summarize_chain(llm=llm, chain_type="stuff", prompt=prompt, verbose=True)
                summary = chain.run(chunks)
                st.write(summary)


            elif text_tech == "LLMChain":
                method = st.sidebar.selectbox("Select a Method", ["Without Prompt", "With Prompt"])

                if method == "Without Prompt":
                    st.write(len(document),"Number of Documents present")
                    doc_no = st.text_input("documnet number")
                    document = (document[int(doc_no)].page_content[:1000])
                    chat_message = [
                        SystemMessage(content="You are an expert in summarizing speech"),
                        HumanMessage(content=f"Please provide a short and concise summary of the following speech:\nText: {document}")
                    ]
                    st.write(llm(chat_message).content)

                elif method == "With Prompt":
                    st.write(len(document),"Number of Documents present")
                    doc_no = st.text_input("documnet number")
                    document = (document[int(doc_no)].page_content[:1000])
                    language = st.sidebar.selectbox("Select a Language", ["English", "Hindi", "Marathi", "German", "French"])
                    generic_temp = """
                        Write a summary of the following speech:
                        Speech: {speech}
                        Translate the precise summary to {language}
                    """
                    prompt = PromptTemplate(input_variables=['speech', 'language'], template=generic_temp)
                    llm_chain = LLMChain(llm=llm, prompt=prompt)
                    summary = llm_chain.run({'speech': document, 'language': language})
                    st.write(summary)

            elif text_tech == "Map-Reduce":
                # Summarization using Map-Reduce
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                text_split = text_splitter.split_documents(document)

                map_prompt_template = """
                    Write a concise and short summary of the following speech:
                    Speech: {text}
                """
                map_prompt = PromptTemplate(input_variables=['text'], template=map_prompt_template)

                final_prompt_template = """
                    Provide the final summary of the entire speech with these important points:
                    Add a motivational title, start the summary with an introduction, and provide a summary in numbered points.
                    Speech: {text}
                """
                final_prompt = PromptTemplate(input_variables=['text'], template=final_prompt_template)

                chain = load_summarize_chain(
                    llm=llm, 
                    chain_type="map_reduce", 
                    map_prompt=map_prompt, 
                    combine_prompt=final_prompt,
                    verbose=True
                )
                summary = chain.run(text_split)
                st.write(summary)

            elif text_tech == "Refine":
                # Summarization using the Refine method
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
                text_split = text_splitter.split_documents(document)

                refine_prompt_template = """
                    Write a concise and short summary of the following speech:
                    Speech: {text}
                """
                refine_prompt = PromptTemplate(input_variables=['text'], template=refine_prompt_template)

                chain = load_summarize_chain(llm=llm, chain_type="refine", verbose=True)
                summary = chain.run(text_split)
                st.write(summary)

        # Clean up the temporary file after processing
        os.remove(temp_file_path)

else:
    st.warning("Please provide the Groq API Key.")




