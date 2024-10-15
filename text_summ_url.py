import validators
import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from dotenv  import load_dotenv
load_dotenv()
# from langchain_huggingface import HuggingFaceEndpoint

# Streamlit app setup
st.set_page_config(page_title="Langchain: Summarize Text from YT or Website", page_icon="🦜")
st.title("Langchain: Summarize Text from YouTube or Website")
st.subheader("Summarize content from a URL")

# Get the API key and URL from user input
with st.sidebar:
    groq_api_key = st.text_input("GROQ_API_KEY", value="", type="password")
    # HF_api_key = st.text_input("Hugging Face token key", value="", type="password")
    # st.header("using Huggingface token")


url = st.text_input("Enter URL (YouTube or Website)", label_visibility="collapsed")

# Initialize the LLM using the Groq API key
llm = ChatGroq(model_name="llama-3.2-3b-preview", groq_api_key=groq_api_key)

# hf_token= os.getenv("HF_TOKEN")
# huggingface model declareation
# repo_id = "mistralai/Mistral-7B-Instruct-v0.3"
# llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=150, temperature=0.7, token=hf_token)


# Define the summarization prompt template
prompt_template = """
**Your task is to create a comprehensive and pedagogical summary of the provided content.** 

**Key guidelines:**

* **Limit the summary to 300 words.**
* **Present the key points in 3 concise bullet points.**
* **Prioritize the most significant information.**
* **Ensure the summary is clear, accurate, objective, and accessible to a student with limited knowledge.**
* **Avoid unnecessary details or jargon.**
* **Use analogies, metaphors, or real-world examples to illustrate complex concepts.**
* **Provide code snippets or examples where appropriate to reinforce understanding.**
* **Conclude with a brief discussion of potential research directions or applications.**
:
{text}
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['text'])

# Summarize button
if st.button("Summarize the content from URL"):
    # Validate input
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the API key and URL to get started.")
    elif not validators.url(url):
        st.error("Please provide a valid URL (either a YouTube video or website URL).")
    else:
        try:
            with st.spinner("Fetching and summarizing content..."):
                # Load the website or YouTube URL content
                if "youtube.com" in url:
                    loader = YoutubeLoader.from_youtube_url(url, add_video_info=True)
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url], 
                        ssl_verify=False, 
                        headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36"}
                    )

                # Load and process the content
                docs = loader.load()

                # Check if the loaded content contains any text
                if not docs or not docs[0].page_content.strip():
                    st.error("The provided URL does not contain any content to summarize.")
                else:
                     # Split the text into smaller chunks
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                    split_docs = text_splitter.split_documents(docs)

                    # Summarization chain
                    chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                    # output_summary = chain.run(docs)
                    # st.success(output_summary)

                     # Summarize each chunk
                    summaries = []
                    for chunk in split_docs:
                        summary = chain.run([chunk])
                        summaries.append(summary)

                    # Join the individual summaries into a single output
                    final_summary = "\n".join(summaries)
                    st.success(final_summary)

        except Exception as e:
            st.exception(f"Exception occurred: {e}")
