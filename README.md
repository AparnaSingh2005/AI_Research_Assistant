# 📚 AI Research Assistant

An AI-powered Research Paper Assistant built with **Streamlit**, **ChromaDB**, **Sentence Transformers**, and **Large Language Models (LLMs)**. The application enables users to upload multiple research papers, perform semantic search, ask questions, generate summaries, analyze trends, compare papers, and gain AI-driven research insights through an interactive dashboard.

---

## 🚀 Features

### 📄 Document Processing

* Upload multiple PDF research papers
* Automatic text extraction and cleaning
* Intelligent text chunking
* Multi-document support

### 🔍 Semantic Search

* Vector embeddings using Sentence Transformers
* ChromaDB vector database
* Semantic similarity search
* Context-aware document retrieval

### 🤖 AI-Powered Assistance

* Question Answering (RAG)
* Research Paper Summarization
* Quiz Generation
* Research Report Generation
* AI Research Advisor
* Key Insights Extraction
* Recommendations

### 📊 Analytics & Visualization

* Paper Statistics
* Metadata Extraction
* Keyword Extraction
* Word Cloud
* Reading Statistics
* Trend Analysis
* Paper Similarity Analysis
* Research Paper Clustering

### 📈 Dashboard

* Overall Statistics
* Interactive Charts
* Research Insights
* Collection Overview
* Downloadable Reports

### 📤 Export Options

* Export Research Reports
* Download Statistics
* Save AI Responses

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Frontend

* Streamlit

### AI & NLP

* Groq API (Llama Models)
* Sentence Transformers
* NLTK

### Vector Database

* ChromaDB

### Data Processing

* Pandas
* NumPy

### Visualization

* Plotly
* Matplotlib
* WordCloud

---

## 📂 Project Structure

```text
AI_Research_Assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── modules/
│   ├── pdf_loader.py
│   ├── text_cleaner.py
│   ├── chunker.py
│   ├── embedding_model.py
│   ├── vector_database.py
│   ├── llm.py
│   ├── paper_statistics.py
│   ├── keyword_extractor.py
│   ├── similarity.py
│   ├── trend_analysis.py
│   ├── clustering.py
│   ├── exporter.py
│   └── ...
│
├── uploads/
└── database/
```

---

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/AparnaSingh2005/AI_Research_Assistant.git

cd AI_Research_Assistant
```

---

### Create a Virtual Environment

#### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file (or configure your deployment secrets) and add your API key.

Example:

```env
GROQ_API_KEY=YOUR_API_KEY
```

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

---

## 📌 Workflow

```text
Upload Research Papers
          │
          ▼
PDF Processing
          │
          ▼
Text Cleaning
          │
          ▼
Chunking
          │
          ▼
Embedding Generation
          │
          ▼
ChromaDB Storage
          │
          ▼
Semantic Search
          │
          ▼
AI Response Generation
          │
          ▼
Visual Analytics & Reports
```

---

## 💡 Key Functionalities

* Multi-PDF Upload
* Retrieval-Augmented Generation (RAG)
* Semantic Search
* Research Summarization
* AI Chat with Papers
* Metadata Extraction
* Keyword Analysis
* Similarity Analysis
* Trend Analysis
* Paper Clustering
* Interactive Dashboard
* Downloadable Reports

---

## 📸 Screenshots

Add screenshots after deployment.

Example:

* Dashboard
* Chat Interface
* Search Results
* Similarity Analysis
* Trend Analysis
* Clustering
* AI Advisor

---

## 🔮 Future Improvements

* User Authentication
* Cloud Database Support
* Citation Generation
* Multi-language Support
* OCR for Scanned PDFs
* Collaborative Research Workspace
* Reference Manager Integration
* Research Paper Recommendation System

---

## 👩‍💻 Author

**Aparna Singh**

B.Tech Information Technology

Interested in Artificial Intelligence, Machine Learning, NLP, and Full-Stack AI Applications.

GitHub: https://github.com/AparnaSingh2005

---

