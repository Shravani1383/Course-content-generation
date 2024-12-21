import os
import streamlit as st
import PyPDF2
from io import BytesIO
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

# Function to list directories in a base directory
def list_directories(base_dir):
    return [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]

# Function to delete pages from a PDF file and save it
def delete_pages_from_pdf(input_pdf_file, pages_to_delete, save_directory, original_file_name):
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

# Function to read PDF text
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PyPDF2.PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Split text into chunks
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks

# Create a FAISS vector store from text chunks
def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Chatbot conversation chain for course content generation
def get_conversational_chain():
    prompt_template = """
    Generate ppt content and transcript for a ppt content for provided topic {question} from {context}. 
    The transcripts should be conversational, in simple words, and explained with examples wherever necessary. 
    Also provide diagrams, image ideas, or any special needs for presentation.
    Slide No. : Title of the slide
    Slide Content: 'Strictly key points'
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

# Clear chat history
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "Provide a topic based on the PDF to generate PPT content and transcript."}
    ]

# Get user input and generate a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")  # type: ignore
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response

# Main function
def main():
    st.set_page_config(page_title="PDF Processing and Course Content Generation", page_icon="📘")

    # Initialize session state for app stages
    if 'stage' not in st.session_state:
        st.session_state.stage = 1  # Stage 1: PDF processing, Stage 2: Course content generation

    # PDF Upload and Page Deletion Logic (Stage 1)
    if st.session_state.stage == 1:
        st.title("Upload and Process PDF")

        base_directory = "directories"
        directories = list_directories(base_directory)
        selected_directory = st.selectbox("Select a directory:", directories)

        if selected_directory:
            selected_directory_path = os.path.join(base_directory, selected_directory)

            # Step 2: Handle file uploader reset and upload PDF file
            if st.session_state.get('uploaded_pdf') is None:
                uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
            else:
                uploaded_pdf = st.session_state['uploaded_pdf']

            if uploaded_pdf is not None:
                st.session_state['uploaded_pdf'] = uploaded_pdf
                original_file_name = uploaded_pdf.name
                pages_input = st.text_input("Enter pages or ranges to delete (e.g., 1, 3, 5-7):")
                save_directory = st.text_input("Enter a directory to save the modified PDF", value=selected_directory_path)

                if st.button("Delete Pages"):
                    if pages_input:
                        pages_to_remove = [int(p) if '-' not in p else p for p in pages_input.split(',')]
                        saved_file_path = delete_pages_from_pdf(st.session_state['uploaded_pdf'], pages_to_remove, save_directory, original_file_name)
                        if saved_file_path:
                            st.success(f"Pages deleted successfully! PDF saved to: {saved_file_path}")
                    else:
                        st.error("Please specify pages to delete.")

            # Next Button to Skip or Move to Next Stage
            if st.button("Next"):
                st.session_state.stage = 2
                st.experimental_rerun()

            # Button for uploading another file (resets the session state)
            if st.button("Upload Another File"):
                st.session_state['uploaded_pdf'] = None  # Reset uploaded PDF
                st.experimental_rerun()

    # Course Content Generation Logic (Stage 2)
    if st.session_state.stage == 2:
        st.title("Generate Course Content From PDF")

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

        st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
        if "messages" not in st.session_state:
            st.session_state.messages = [{"role": "assistant", "content": "Upload some PDFs and ask me a question."}]

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
                st.session_state.messages.append({"role": "assistant", "content": full_response})

if __name__ == "__main__":
    main()
