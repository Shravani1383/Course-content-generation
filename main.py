import os
import streamlit as st
import PyPDF2
from io import BytesIO
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# App 1 logic: Delete Pages from a PDF
def app1():
    st.title("PDF Page Deletion App")

    # Base directory where subdirectories are located
    base_directory = "directories"  # Change this to your actual base directory

    # Step 1: Select a subdirectory
    directories = list_directories(base_directory)
    selected_directory = st.selectbox("Select a directory:", directories)

    # Ensure a directory is selected before proceeding
    if selected_directory:
        # Full path to the selected directory
        selected_directory_path = os.path.join(base_directory, selected_directory)

        # Initialize session state for file reset handling
        if 'uploaded_pdf' not in st.session_state:
            st.session_state['uploaded_pdf'] = None

        # Step 2: Handle file uploader and manage session state for uploaded PDF
        if st.session_state['uploaded_pdf'] is None:
            uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
        else:
            uploaded_pdf = st.session_state['uploaded_pdf']

        if uploaded_pdf is not None:
            # Store the uploaded PDF in session state
            st.session_state['uploaded_pdf'] = uploaded_pdf
            original_file_name = uploaded_pdf.name

            # Step 3: Enter pages or ranges to delete
            pages_input = st.text_input("Enter pages or ranges to delete (e.g., 1, 3, 5-7):")

            # Step 4: Provide a default save directory
            save_directory = st.text_input("Enter a directory to save the modified PDF", value=selected_directory_path)

            # Step 5: Process when user clicks "Delete Pages"
            if st.button("Delete Pages"):
                if pages_input:
                    try:
                        # Parse the input for single pages and ranges
                        pages_to_remove = []
                        for part in pages_input.split(','):
                            part = part.strip()
                            if '-' in part:
                                pages_to_remove.append(part)  # Keep range as 'start-end'
                            else:
                                pages_to_remove.append(int(part))  # Single pages as integers

                        # Call the function to delete pages and save the file
                        saved_file_path = delete_pages_from_pdf(uploaded_pdf, pages_to_remove, save_directory, original_file_name)

                        if saved_file_path:
                            st.success(f"Pages deleted successfully! PDF saved to: {saved_file_path}")
                    except Exception as e:
                        st.error(f"Invalid input or error processing PDF: {e}")
                else:
                    st.error("Please specify pages to delete.")

        # Step 6: Provide option to upload another file (resets session state)
        if st.button("Upload Another File"):
            # Reset the session state and rerun the app to clear the uploaded file
            st.session_state['uploaded_pdf'] = None
            st.experimental_rerun()

# App 2 logic: Generate PPT Content from PDF
def app2():
    st.title("Generate PPT Content From Books")
    
    base_dir = "directories"
    available_dirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    selected_dir = st.selectbox("Select a directory", available_dirs)

    if selected_dir:
        dir_path = os.path.join(base_dir, selected_dir)
        pdf_files = [os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith('.pdf')]

        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_files)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    st.write("Upload PDFs and ask a question to generate PPT content and transcript.")

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask a question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = user_input(prompt)
                    placeholder = st.empty()
                    full_response = ''
                    for item in response['output_text']:
                        full_response += item
                        placeholder.markdown(full_response)
                    placeholder.markdown(full_response)
                if response is not None:
                    message = {"role": "assistant", "content": full_response}
                    st.session_state.messages.append(message)

# Helper functions
def list_directories(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

def delete_pages_from_pdf(input_pdf_file, pages_to_delete, save_directory, original_file_name):
    # Your logic for deleting pages from PDF
    try:
        pages_to_delete_set = set()
        for item in pages_to_delete:
            if isinstance(item, int):  # Single page numbers
                pages_to_delete_set.add(item - 1)  # Convert to 0-indexed
            elif isinstance(item, str) and '-' in item:  # Page ranges
                start, end = map(int, item.split('-'))
                pages_to_delete_set.update(range(start - 1, end))  # 0-indexed

        reader = PyPDF2.PdfReader(input_pdf_file)
        writer = PyPDF2.PdfWriter()

        for page_num in range(len(reader.pages)):
            if page_num not in pages_to_delete_set:
                writer.add_page(reader.pages[page_num])

        os.makedirs(save_directory, exist_ok=True)
        base_name, ext = os.path.splitext(original_file_name)
        modified_file_name = f"{base_name}_modified{ext}"
        output_file_path = os.path.join(save_directory, modified_file_name)

        with open(output_file_path, "wb") as output_pdf_file:
            writer.write(output_pdf_file)

        return output_file_path

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

def get_conversational_chain():
    prompt_template = """
    Generate ppt content and transcript for a ppt content for provided topic {question} from {context} provide, the transcripts should be like conversation and in 
    easy words, and explain it with example wherever necessary also provide diagrams, image ideas or any special need for presentation.
    Slide No. : Title of the slide
    Slide Content:
    'Strictly key points'
    Transcript:
    Image Ideas:
    
    Context:\n {context}?\n
    Question: \n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", client=genai, temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain

# Main App
def main():
    st.sidebar.title("Select App")
    app_selection = st.sidebar.selectbox("Choose an app", ("PDF Page Deletion", "Generate PPT Content"))

    if app_selection == "PDF Page Deletion":
        app1()
    elif app_selection == "Generate PPT Content":
        app2()

if __name__ == "__main__":
    main()
