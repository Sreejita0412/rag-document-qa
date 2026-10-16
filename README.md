# 📚 RAG Document Q&A Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents, ask questions about their content, and receive AI-generated answers with relevant PDF page citations.

## ✨ Features

- **PDF Upload:** Upload a PDF and ask questions about its content.
- **Document Processing:** Extract text from PDFs and split it into manageable chunks.
- **Semantic Search:** Retrieve relevant passages using Hugging Face embeddings and Chroma.
- **AI-Powered Answers:** Use Google Gemini to generate answers based on retrieved document passages.
- **Page Citations:** Display PDF page references and retrieved passages to help verify answers.
- **Multiple Documents:** Switch between an uploaded PDF and the original UAV research paper.
- **Retrieval Evaluation:** Evaluate retrieval performance using questions with expected source pages.

## 🛠️ Tech Stack

- **Language:** Python
- **Frontend:** Streamlit
- **LLM:** Google Gemini
- **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
- **Vector Database:** Chroma
- **Document Processing:** PyPDFLoader and LangChain
- **Evaluation:** Python-based retrieval evaluation

## ⚙️ How It Works

1. A user uploads a PDF or selects the original research paper.
2. The application extracts text from the document.
3. The extracted text is divided into overlapping chunks.
4. The embedding model converts chunks into numerical vectors.
5. Chroma retrieves passages relevant to the user's question.
6. Gemini generates an answer using the retrieved context.
7. The application displays the answer alongside supporting PDF page references.

## 📁 Project Structure

```text
rag-document-qa/
├── app.py
├── requirements.txt
├── .gitignore
├── src/
│   ├── document_loader.py
│   ├── embeddings.py
│   ├── test_splitter.py
│   └── test_vector_store.py
├── evaluation/
│   ├── evaluate_retrieval.py
│   └── questions.json
├── data/
│   └── sample.pdf
├── test_gemini.py
├── test_rag.py
└── test_retrieval.py
```

*Note: The exact files in `src/` and `evaluation/` may vary as the project evolves. The locally generated Chroma database and environment files are excluded from version control.*

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or a compatible version for the installed dependencies
- Git
- A Google Gemini API key

### 1. Clone the repository

```bash
git clone https://github.com/Sreejita0412/rag-document-qa.git
cd rag-document-qa
```

### 2. Create and activate a virtual environment

**Windows PowerShell**

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Configure the API key

Create a file named `.env` in the project root and add:

```text
GOOGLE_API_KEY=your_gemini_api_key_here
```

Replace the placeholder with your own API key. **Never commit or share your `.env` file.**

### 5. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal. Upload a PDF or use the original UAV research paper, then enter a question in the chat interface.

## 📊 Retrieval Evaluation

The retrieval evaluation uses a set of questions paired with expected source pages to measure whether relevant document passages are retrieved.

In the current test run, the system achieved a **96% retrieval hit rate (24 out of 25 questions)**.

This metric measures retrieval success, not the factual accuracy of every generated answer.

## 🔒 Security and Data Handling

- API keys and environment variables are excluded through `.gitignore`.
- The local virtual environment is not included in the repository.
- The generated Chroma database is excluded and can be rebuilt as needed.
- Do not upload confidential or sensitive documents to the application unless you have permission to process them.

## 🔮 Future Improvements

- Measure and optimize retrieval and answer-generation latency.
- Improve retrieval quality and evaluate generated-answer correctness.
- Add support for additional document formats.
- Improve error handling and document-processing feedback.
- Deploy the application for public access.

## 👩‍💻 Author

**Sreejita Sarkar**

Information Technology student interested in AI, Machine Learning, and Retrieval-Augmented Generation.

---

*Built as a hands-on project to learn document processing, semantic search, vector databases, and LLM-based question answering.*
