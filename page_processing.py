import os
import streamlit as st
import PyPDF2
from io import BytesIO

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

# Base directory where all subdirectories are located
base_directory = "directories"  # Change this to your actual base directory

# Initialize session state for uploaded file reset
if 'uploaded_pdf' not in st.session_state:
    st.session_state['uploaded_pdf'] = None

# Step 1: Select a subdirectory
directories = list_directories(base_directory)
selected_directory = st.selectbox("Select a directory:", directories)

if selected_directory:
    # Full path to the selected directory
    selected_directory_path = os.path.join(base_directory, selected_directory)

    # Step 2: Handle file uploader reset and upload PDF file
    if st.session_state['uploaded_pdf'] is None:
        uploaded_pdf = st.file_uploader("Upload a PDF file", type="pdf")
    else:
        uploaded_pdf = st.session_state['uploaded_pdf']

    # If a new file is uploaded, store it in session state
    if uploaded_pdf is not None:
        st.session_state['uploaded_pdf'] = uploaded_pdf

    # Use the stored uploaded file from session state
    if st.session_state['uploaded_pdf'] is not None:
        # Extract the original file name
        original_file_name = uploaded_pdf.name

        # Step 3: Enter pages or ranges to delete
        pages_input = st.text_input("Enter pages or ranges to delete (e.g., 1, 3, 5-7):")

        # Step 4: Choose a save directory (default is the selected directory)
        save_directory = st.text_input("Enter a directory to save the modified PDF", value=selected_directory_path)

        # Step 5: Process when user clicks "Delete Pages"
        if st.button("Delete Pages"):
            if pages_input:
                try:
                    # Parse the input and convert to list of integers or ranges
                    pages_to_remove = []
                    for part in pages_input.split(','):
                        part = part.strip()
                        if '-' in part:
                            pages_to_remove.append(part)  # Keep range as 'start-end'
                        else:
                            pages_to_remove.append(int(part))  # Single pages to integers

                    # Process the PDF
                    saved_file_path = delete_pages_from_pdf(st.session_state['uploaded_pdf'], pages_to_remove, save_directory, original_file_name)

                    # Step 6: Confirm the PDF was saved
                    if saved_file_path:
                        st.success(f"Pages deleted successfully! PDF saved to: {saved_file_path}")
                except Exception as e:
                    st.error(f"Invalid input or error processing PDF: {e}")
            else:
                st.error("Please specify pages to delete.")

        # Step 7: Option to upload another file (resets session state)
        if st.button("Upload Another File"):
            st.session_state['uploaded_pdf'] = None  # Clear the stored file in session state
            st.experimental_rerun()  # Rerun the app to reset the uploader
