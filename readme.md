# 📚 **E-Book to Course Content Generator**

Transform e-books into professional course content with this powerful tool! This project lets you upload e-books (in PDF format), process them to generate slide decks (PPT) along with transcripts, and even perform additional tasks like deleting specific pages from PDFs.

---

## 🚀 **Features**
- Generate **PPT Content** with key points, conversational transcripts, and slide design ideas.
- Delete specific pages from PDFs with a user-friendly interface.
- Store e-book content in vector embeddings for **efficient searching and processing**.

---

## 🛠️ **Installation**

### 1️⃣ Clone the Repository
```bash
git clone <repository_url>
cd <repository_folder>
```

### 2️⃣ Install Requirements
Ensure you have Python 3.11 or above installed. Install dependencies by running:
```bash
pip install -r requirements.txt
```

---

## 📄 **Setup**

### 1️⃣ Create `.env` File
Create a `.env` file in the root directory and add your **Google API key**:
```
GOOGLE_API_KEY=<your_google_api_key>
```

### 2️⃣ Obtain a Google API Key
- Visit the [Google Cloud Console](https://console.cloud.google.com/).
- Create a project and enable the **Generative AI API**.
- Generate an API key and add it to your `.env` file.

---

## ▶️ **Usage**

### 1️⃣ Run the Application
Start the Streamlit app:
```bash
streamlit run main.py
```

### 2️⃣ Choose a Functionality
On the app sidebar, select one of the following options:
1. **PDF Page Deletion**  
   Upload a PDF, specify pages/ranges to delete, and save the modified PDF.
2. **Generate PPT Content**  
   Upload PDFs, process them to generate course content including PPT slides and conversational transcripts.

---

## 🛠️ **Code Structure**

### Main Functions
- **`app1`**: Handles the PDF Page Deletion logic.
- **`app2`**: Processes PDFs to extract text, generate slide content, and create transcripts.
- **`list_directories`**: Lists subdirectories for file management.
- **`delete_pages_from_pdf`**: Deletes specified pages from a PDF.
- **`get_pdf_text`**: Extracts text from uploaded PDFs.
- **`get_text_chunks`**: Splits text into manageable chunks for vector storage.
- **`get_vector_store`**: Creates a FAISS vector store for efficient querying.
- **`get_conversational_chain`**: Generates conversational and course-ready PPT content.

---

## 📂 **Project Folder Structure**
```
project/
│
├── main.py               # Main Streamlit app
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables
├── directories/          # Directory to store user files
│   ├── <subdirectory_1>/
│   ├── <subdirectory_2>/
│   └── ...
```

---

## ✨ **Future Enhancements**
- Support for additional file formats (e.g., Word, EPUB).
- Add more AI models for slide generation.
- Multi-user support with authentication.

---

## 📝 **License**
This project is licensed under the MIT License. See the `LICENSE` file for more details.

---

## 💡 **Contributions**
Contributions are welcome! Feel free to fork the repo and submit pull requests for new features, bug fixes, or improvements.

---

Enjoy turning your e-books into impactful presentations! 🎉
