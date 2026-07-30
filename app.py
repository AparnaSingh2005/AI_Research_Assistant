import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import seaborn as sns
from datetime import datetime
from modules.pdf_loader import PDFLoader
from modules.text_cleaner import TextCleaner
from modules.chunker import TextChunker
from modules.embedding_model import EmbeddingModel
from modules.vector_database import VectorDatabase
from modules.llm import LLM
from modules.paper_statistics import PaperStatistics
from modules.keyword_extractor import KeywordExtractor
from modules.exporter import Exporter
from collections import Counter
from wordcloud import STOPWORDS, WordCloud
from modules.trend_analysis import TrendAnalysis
from modules.similarity import PaperSimilarity
from modules.clustering import PaperClustering
import io


import nltk

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

# -------------------------------
# PAGE CONFIG
# -------------------------------

st.set_page_config(
    page_title="AI Research Paper Assistant",
    page_icon="📚",
    layout="wide"
)

# -------------------------------
# CREATE DIRECTORIES
# -------------------------------

os.makedirs("uploads", exist_ok=True)
os.makedirs("database", exist_ok=True)

# -------------------------------
# CACHE MODELS
# -------------------------------

@st.cache_resource
def load_embedder():
    return EmbeddingModel()


@st.cache_resource
def load_llm():
    return LLM()


embedder = load_embedder()
llm = load_llm()
exporter = Exporter()
similarity = PaperSimilarity()
trend = TrendAnalysis()
clustering = PaperClustering()


keyword_extractor = KeywordExtractor()
# -------------------------------
# SESSION STATE
# -------------------------------
import pandas as pd

if "similarity_df" not in st.session_state:
    st.session_state.similarity_df = pd.DataFrame()

if "database" not in st.session_state:
    st.session_state.database = VectorDatabase()

if "database_ready" not in st.session_state:
    st.session_state.database_ready = False

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = []

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "statistics" not in st.session_state:
    st.session_state.statistics = {}

if "metadata" not in st.session_state:
    st.session_state.metadata = ""

if "metadata_dict" not in st.session_state:
    st.session_state.metadata_dict = {}

if "keywords" not in st.session_state:
    st.session_state.keywords = {}

if "citations" not in st.session_state:
    st.session_state.citations = ""

# -------------------------------
# GLOBAL VARIABLES
# -------------------------------

text = ""
words = []
counter = Counter()
keyword_df = pd.DataFrame(
    columns=["Keyword", "Frequency"]
)
top10 = pd.DataFrame(
    columns=["Keyword", "Frequency"]
)
documents = st.session_state.get("documents", [])
# -------------------------------
# APP SETTINGS
# -------------------------------

if "top_k" not in st.session_state:
    st.session_state.top_k = 5

if "temperature" not in st.session_state:
    st.session_state.temperature = 0.3

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

if "show_sources" not in st.session_state:
    st.session_state.show_sources = True

if "selected_model" not in st.session_state:
    st.session_state.selected_model = "llama-3.1-8b-instant"

if "chat_history_enabled" not in st.session_state:
    st.session_state.chat_history_enabled = True


# -------------------------------
# TITLE
# -------------------------------

st.title("📚 AI Research Paper Assistant")

st.markdown("""
Upload one or more research papers and interact with them using
Retrieval-Augmented Generation (RAG).

Features

- Multi PDF Upload
- Semantic Search
- Gemini AI
- Research Summary
- Quiz Generation
- Research Gap Detection
- Recommendations
""")

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("📚 AI Research Dashboard")

# ----------------------------------------
# Status
# ----------------------------------------

status = (
    "🟢 Ready"
    if st.session_state.database_ready
    else "🟡 Waiting for PDFs"
)

st.sidebar.success(status)

# ----------------------------------------
# Metrics
# ----------------------------------------

st.sidebar.metric(
    "📚 Stored Chunks",
    st.session_state.database.count()
)

st.sidebar.metric(
    "❓ Questions Asked",
    len(
        [
            msg for msg in st.session_state.messages
            if msg["role"] == "user"
        ]
    )
)

paper_count = len(
    {
        chunk["source"]
        for chunk in st.session_state.chunks
    }
)

st.sidebar.metric(
    "📄 Research Papers",
    paper_count
)

st.sidebar.divider()

# =====================================================
# SETTINGS
# =====================================================

st.sidebar.header("⚙ Retrieval Settings")

# Number of retrieved chunks
st.session_state.top_k = st.sidebar.slider(
    "Top K Results",
    min_value=1,
    max_value=10,
    value=st.session_state.top_k
)

# Gemini temperature
st.session_state.temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.0,
    max_value=1.0,
    value=st.session_state.temperature,
    step=0.1
)

st.sidebar.divider()

# ======================================================
# LLM Model Selection
# ======================================================

st.sidebar.header("🤖 Groq Model")

models = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant"
]

# Initialize session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = models[0]

# Sidebar model selector
selected_model = st.sidebar.selectbox(
    "Choose Model",
    models,
    index=models.index(st.session_state.selected_model)
)

# Save selection
st.session_state.selected_model = selected_model

# Update LLM
llm.set_model(selected_model)

st.sidebar.divider()

# =====================================================
# CHAT OPTIONS
# =====================================================

st.sidebar.header("💬 Chat")

st.session_state.show_sources = st.sidebar.checkbox(
    "Show Sources",
    value=st.session_state.show_sources
)

st.session_state.chat_history_enabled = st.sidebar.checkbox(
    "Save Chat History",
    value=st.session_state.chat_history_enabled
)

st.sidebar.divider()

# =====================================================
# ACTIONS
# =====================================================

st.sidebar.header("🛠 Actions")

if st.sidebar.button(
    "🗑 Clear Chat",
    use_container_width=True
):

    st.session_state.messages = []

    st.rerun()


if st.sidebar.button(
    "🗑 Clear Database",
    use_container_width=True
):

    st.session_state.database.clear_database()

    st.session_state.database_ready = False

    st.session_state.documents = []

    st.session_state.chunks = []

    st.session_state.statistics = {}

    st.session_state.messages = []

    st.success("Database Cleared Successfully")

    st.rerun()

st.sidebar.divider()

# =====================================================
# ABOUT
# =====================================================

st.sidebar.header("ℹ About")

st.sidebar.info(
    """
AI Research Paper Assistant

✅ Multi-PDF Support

✅ Semantic Search

✅ Gemini AI

✅ Research Gap Detection

✅ Paper Comparison

✅ Analytics Dashboard

Built using

• Streamlit

• ChromaDB

• Sentence Transformers

• Gemini AI
"""
)

# -------------------------------
# TABS
# -------------------------------
tab_upload, \
tab_chat, \
tab_summary, \
tab_quiz, \
tab_gap, \
tab_recommend, \
tab_compare, \
tab_stats, \
tab_similarity, \
tab_compare_ai,\
tab_trends, \
tab_cluster,\
tab_advisor,\
tab_search,\
tab_dashboard,\
tab_keywords, \
tab_metadata, \
tab_citation, \
tab_info, \
tab_search_2 = st.tabs(

[
    "📄 Upload",
    "💬 Chat",
    "📝 Summary",
    "🎯 Quiz",
    "🔬 Research Gap",
    "💡 Recommendations",
    "⚖ Compare",
    "📊 Statistics",
    "📊 Similarity",
    "🤖 AI Compare",
    "📈 Trends",
    "🧩 Clustering",
    "🎓 AI Advisor",
    "🔍 Search",
    "🏠 Dashboard",
    "🏷 Keywords",
    "📑 Metadata",
    "📚 Citations",
    "ℹ Paper Info",
    "🔍 Advanced Search"

]
)


# =====================================================
# UPLOAD TAB
# =====================================================

with tab_upload:

    st.header("Upload Research Papers")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )

    if st.button("Process PDFs"):

        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
            st.stop()

        all_documents = []
        all_chunks = []

        cleaner = TextCleaner()
        chunker = TextChunker()

        progress = st.progress(0)

        total = len(uploaded_files)

        for index, uploaded_file in enumerate(uploaded_files):

            save_path = os.path.join(
                "uploads",
                uploaded_file.name
            )

            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            with st.spinner(f"Reading {uploaded_file.name}"):

                loader = PDFLoader(save_path)

                # Extract complete text
                extracted_text, total_pages = loader.extract_text()

                # Extract page-wise data
                pages = loader.extract_pages()

         

                # Clean text
                cleaned = cleaner.clean_text(extracted_text)

                # Create chunks
                chunks = chunker.create_chunks(pages)
                

            all_documents.append(
                {
                    "name": uploaded_file.name,
                    "text": cleaned,
                    "pages": pages
                }
            )

            paper_keywords = keyword_extractor.extract(
                cleaned,
                top_n=10
            )

            st.session_state.keywords[
                uploaded_file.name
            ] = paper_keywords





            all_chunks.extend(chunks)

            progress.progress(
                (index + 1) / total
            )

        # ----------------------------------------
        # Generate Embeddings
        # ----------------------------------------

        with st.spinner("Generating Embeddings..."):

            embeddings = embedder.embed_documents(
                all_chunks
            )

        # ----------------------------------------
        # Store in ChromaDB
        # ----------------------------------------

        with st.spinner("Creating Vector Database..."):

            database = st.session_state.database

            database.clear_database()

            database.add_documents(
                all_chunks,
                embeddings
            )

        # ----------------------------------------
        # Save Session State
        # ----------------------------------------

        st.session_state.documents = all_documents

        st.session_state.chunks = all_chunks

        st.session_state.database_ready = True

        # ----------------------------------------
        # Statistics
        # ----------------------------------------

        stats = PaperStatistics()

        total_words = 0

        total_characters = 0

        total_pages = 0

        total_paragraphs = 0

        for document in all_documents:

            paper_stats = stats.statistics(
                document["text"]
            )

            total_words += paper_stats["Words"]

            total_characters += paper_stats["Characters"]

            total_pages += len(
                document["pages"]
            )

            total_paragraphs += paper_stats["Paragraphs"]

        st.session_state.statistics = {

            "papers": len(all_documents),

            "chunks": len(all_chunks),

            "pages": total_pages,

            "words": total_words,

            "characters": total_characters,

            "paragraphs": total_paragraphs,

            "embedding_dimension":
            embedder.embedding_dimension()

        }

        st.success("Research Papers Processed Successfully!")

        st.balloons()

        st.divider()

        st.subheader("Uploaded Papers")

        dataframe = pd.DataFrame(

            [
                {

                    "Paper": doc["name"],

                    "Pages": len(doc["pages"]),

                    "Characters": len(doc["text"]),

                    "Words": len(doc["text"].split())

                }

                for doc in all_documents

            ]

        )

        st.dataframe(
            dataframe,
            use_container_width=True
        )

# =====================================================
# CHAT TAB
# =====================================================
# =====================================================
# CHAT TAB
# =====================================================

with tab_chat:

    st.header("Ask Questions")

    if not st.session_state.database_ready:

        st.info(
            "Please upload research papers first."
        )

    else:

        # Display Chat History
        for message in st.session_state.messages:

            with st.chat_message(
                message["role"]
            ):

                st.markdown(
                    message["content"]
                )

        # Chat Input
        question = st.chat_input(
            "Ask about your research papers..."
        )

        if question:

            # ----------------------------------------
            # Store User Message
            # ----------------------------------------
            if st.session_state.chat_history_enabled:
                st.session_state.messages.append(
                    {
                        "role": "user",
                        "content": question
                    }
                )

            with st.chat_message("user"):

                st.markdown(question)

            # ----------------------------------------
            # Search Vector DB
            # ----------------------------------------

            with st.spinner(
                "Searching Research Papers..."
            ):

                query_embedding = embedder.embed_query(
                    question
                )

                results = st.session_state.database.search(
                    query_embedding,
                    top_k=st.session_state.top_k
                )

            # ----------------------------------------
            # Build Context
            # ----------------------------------------

            context = ""

            for item in results:

                meta = item["metadata"]

                context += f"""
Paper: {meta['source']}

Page: {meta['page']}

Chunk: {meta['chunk_id']}

Similarity Score: {round(item['similarity'] * 100, 2)}%

Content:
{item['document']}

--------------------------------------------------
"""

            # ----------------------------------------
            # Gemini Response
            # ----------------------------------------

            with st.spinner(
                "Generating AI Response..."
            ):

                try:
                    llm.set_model(
                        st.session_state.selected_model
                 )
                    answer = llm.ask(
                        question,
                        context
                    )

                except Exception as e:

                    answer = f"Error: {str(e)}"

            # ----------------------------------------
            # Assistant Response
            # ----------------------------------------

            with st.chat_message(
                "assistant"
            ):

                st.markdown(answer)

                # ----------------------------
                # Best Match
                # ----------------------------

                if results:

                    best = results[0]

                    st.success(
                        f"🎯 Best Match: "
                        f"{best['metadata']['source']} "
                        f"({best['similarity']:.2%})"
                    )

                # ----------------------------
                # Sources Used
                # ----------------------------
                if results and st.session_state.show_sources:

                    with st.expander("📚 Sources Used"):


                        for i, item in enumerate(
                            results,
                            start=1
                        ):

                            meta = item["metadata"]

                            similarity = round(
                                item["similarity"] * 100,
                                2
                            )

                            st.markdown(
                                f"""
### Source {i}

📄 **Paper:** {meta["source"]}

📑 **Page:** {meta["page"]}

🧩 **Chunk:** {meta["chunk_id"]}

🎯 **Similarity:** {similarity}%
---
"""
                            )

                # ----------------------------
                # Similarity Ranking
                # ----------------------------

                if results:

                    st.subheader(
                        "📊 Similarity Ranking"
                    )

                    ranking = pd.DataFrame(

                        [
                            {
                                "Rank": i + 1,
                                "Paper": item["metadata"]["source"],
                                "Page": item["metadata"]["page"],
                                "Chunk": item["metadata"]["chunk_id"],
                                "Similarity (%)": round(
                                    item["similarity"] * 100,
                                    2
                                )
                            }

                            for i, item in enumerate(results)
                        ]

                    )

                    st.dataframe(
                        ranking,
                        use_container_width=True,
                        hide_index=True
                    )

            # ----------------------------------------
            # Save Assistant Response
            # ----------------------------------------

            if st.session_state.chat_history_enabled:

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )


        # ----------------------------------------
        # Download Chat History
        # ----------------------------------------

        if st.session_state.messages:

            history = ""

            for msg in st.session_state.messages:

                history += (
                    f"{msg['role'].upper()}\n"
                    f"{msg['content']}\n\n"
                )

            st.download_button(
                "⬇ Download Chat",
                history,
                file_name="chat_history.txt",
                mime="text/plain"
            )

        
chat_history = ""

for message in st.session_state.messages:

    chat_history += (
        f"{message['role'].upper()}\n"
        f"{message['content']}\n\n"
    )

st.subheader("Export Chat")

format = st.selectbox(
    "Format",
    ["TXT", "DOCX", "PDF"],
    key="chat_export"
)

if st.button("Export Chat"):

    if format == "TXT":
        filename = exporter.export_txt(
            chat_history,
            "chat_history.txt"
        )

    elif format == "DOCX":
        filename = exporter.export_docx(
            chat_history,
            "chat_history.docx"
        )

    else:
        filename = exporter.export_pdf(
            chat_history,
            "chat_history.pdf"
        )

    with open(filename, "rb") as file:

        st.download_button(
            "⬇ Download Export",
            file,
            file_name=filename
        )

# =====================================================
# SUMMARY TAB
# =====================================================

with tab_summary:

    st.header("📄 Research Paper Summary")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Generate Summary"):

            with st.spinner("Generating Summary..."):

                context = ""

                for document in st.session_state.documents:

                    context += document["text"] + "\n\n"

                try:

                    summary = llm.summarize(context)

                    st.success("Summary Generated Successfully!")

                    st.markdown(summary)

                    st.subheader("Export Summary")

                    format = st.selectbox(
                        "Format",
                        [
                            "TXT",
                            "DOCX",
                            "PDF"
                        ],
                        key="summary_export"
                    )

                    if st.button("Export Summary"):

                        if format == "TXT":

                            filename = exporter.export_txt(
                                summary,
                                "summary.txt"
                            )

                        elif format == "DOCX":

                            filename = exporter.export_docx(
                                summary,
                                "summary.docx"
                            )

                        else:

                            filename = exporter.export_pdf(
                                summary,
                                "summary.pdf"
                            )

                        with open(filename, "rb") as file:

                            st.download_button(
                                "⬇ Download",
                                file,
                                file_name=filename
                            )

                    st.download_button(

                        "⬇ Download Summary",

                        summary,

                        file_name="research_summary.txt",

                        mime="text/plain"

                    )

                except Exception as e:

                    st.error(f"Error: {e}")

# =====================================================
# QUIZ TAB
# =====================================================

with tab_quiz:

    st.header("🎯 AI Quiz Generator")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Generate Quiz"):

            with st.spinner("Creating Quiz..."):

                context = ""

                for document in st.session_state.documents:

                    context += document["text"] + "\n\n"

                try:

                    quiz = llm.generate_quiz(
                        context
                    )

                    st.success(
                        "Quiz Generated Successfully!"
                    )

                    st.markdown(quiz)
                    st.subheader("Export Quiz")

                    format = st.selectbox(
                        "Format",
                        ["TXT", "DOCX", "PDF"],
                        key="quiz_export"
                    )

                    if st.button("Export Quiz"):

                        if format == "TXT":
                            filename = exporter.export_txt(quiz, "quiz.txt")

                        elif format == "DOCX":
                            filename = exporter.export_docx(quiz, "quiz.docx")

                        else:
                            filename = exporter.export_pdf(quiz, "quiz.pdf")

                        with open(filename, "rb") as file:

                            st.download_button(
                                "⬇ Download",
                                file,
                                file_name=filename
                            )

                    

                except Exception as e:

                    st.error(
                        f"Error generating quiz: {e}"
                    )

# =====================================================
# RESEARCH GAP TAB
# =====================================================

with tab_gap:

    st.header("🔬 Research Gap Analysis")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Find Research Gaps"):

            with st.spinner("Analyzing Research Papers..."):

                context = ""

                for document in st.session_state.documents:

                    context += document["text"]
                    context += "\n\n"

                try:

                    research_gap = llm.research_gap(
                        context
                    )

                    st.success(
                        "Research Gap Analysis Completed!"
                    )

                    st.markdown(research_gap)

                    st.download_button(

                        "⬇ Download Research Gap",

                        research_gap,

                        file_name="research_gap.txt",

                        mime="text/plain"

                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

# =====================================================
# RECOMMENDATIONS TAB
# =====================================================

with tab_recommend:

    st.header("💡 AI Recommendations")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Generate Recommendations"):

            with st.spinner("Generating Recommendations..."):

                context = ""

                for document in st.session_state.documents:

                    context += document["text"]
                    context += "\n\n"

                try:

                    recommendations = llm.recommendations(
                        context
                    )

                    st.success(
                        "Recommendations Generated Successfully!"
                    )

                    st.markdown(recommendations)

                    st.subheader("Export Recommendations")

                    format = st.selectbox(
                        "Format",
                        ["TXT", "DOCX", "PDF"],
                        key="recommend_export"
                    )

                    if st.button("Export Recommendations"):

                        if format == "TXT":
                            filename = exporter.export_txt(
                                recommendations,
                                "recommendations.txt"
                            )
        
                        elif format == "DOCX":
                            filename = exporter.export_docx(
                            recommendations,
                            "recommendations.docx"
                            )

                        else:
                            filename = exporter.export_pdf(
                            recommendations,
                            "recommendations.pdf"
                        )

                        with open(filename, "rb") as file:

                             st.download_button(
                                "⬇ Download",
                                file,
                                file_name=filename
                            )



                    st.download_button(

                        "⬇ Download Recommendations",

                        recommendations,

                        file_name="recommendations.txt",

                        mime="text/plain"

                    )

                except Exception as e:

                    st.error(
                        f"Error: {e}"
                    )

# =====================================================
# SIMILARITY TAB
# =====================================================

with tab_similarity:

    st.header("📊 Paper Similarity Analysis")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    elif len(st.session_state.documents) < 2:

        st.warning("Upload at least two research papers.")

    else:

        if st.button("🚀 Generate Similarity Matrix"):

            with st.spinner("Calculating Similarity..."):

                st.session_state.similarity_df = similarity.build_matrix(
                    st.session_state.documents,
                    embedder
                )

                st.success("✅ Similarity Matrix Generated!")

        # ------------------------------------------------
        # Display Results
        # ------------------------------------------------

        if not st.session_state.similarity_df.empty:

            similarity_df = st.session_state.similarity_df

            # --------------------------------------------
            # Similarity Matrix
            # --------------------------------------------

            st.subheader("📋 Similarity Matrix")

            st.dataframe(
                similarity_df.style.format("{:.2f}"),
                use_container_width=True
            )

            # --------------------------------------------
            # Most Similar
            # --------------------------------------------

            pair, score = similarity.most_similar(similarity_df)

            st.success(f"""
### 🏆 Most Similar Papers

**{pair[0]}**

⬄

**{pair[1]}**

Similarity Score: **{score:.2%}**
""")

            # --------------------------------------------
            # Least Similar
            # --------------------------------------------

            pair, score = similarity.least_similar(similarity_df)

            st.warning(f"""
### 📉 Least Similar Papers

**{pair[0]}**

⬄

**{pair[1]}**

Similarity Score: **{score:.2%}**
""")

            # --------------------------------------------
            # Download CSV
            # --------------------------------------------

            csv = similarity_df.to_csv(index=True)

            st.download_button(
                "⬇ Download Similarity Matrix",
                csv,
                file_name="similarity_matrix.csv",
                mime="text/csv"
            )

            # --------------------------------------------
            # Heatmap
            # --------------------------------------------

            st.subheader("🔥 Similarity Heatmap")

            fig, ax = plt.subplots(figsize=(8, 6))

            sns.heatmap(
                similarity_df,
                annot=True,
                cmap="YlGnBu",
                fmt=".2f",
                linewidths=0.5,
                square=True,
                ax=ax
            )

            plt.tight_layout()

            st.pyplot(fig)

            plt.close(fig)

            # --------------------------------------------
            # Similarity Ranking
            # --------------------------------------------

            st.subheader("🏆 Paper Similarity Ranking")

            ranking = []

            papers = list(similarity_df.columns)

            for i in range(len(papers)):
                for j in range(i + 1, len(papers)):
                    ranking.append({

                        "Paper 1": papers[i],

                        "Paper 2": papers[j],

                        "Similarity (%)": round(
                            similarity_df.iloc[i, j] * 100,
                            2
                        )

                    })

            ranking_df = pd.DataFrame(ranking)

            if not ranking_df.empty:

                ranking_df = ranking_df.sort_values(
                    "Similarity (%)",
                    ascending=False
                )

                st.dataframe(
                    ranking_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ----------------------------------------
                # Similarity Chart
                # ----------------------------------------

                st.subheader("📊 Similarity Scores")

                fig2, ax2 = plt.subplots(figsize=(10, 5))

                labels = [
                    f"{row['Paper 1']} ↔ {row['Paper 2']}"
                    for _, row in ranking_df.iterrows()
                ]

                scores = ranking_df["Similarity (%)"]

                ax2.barh(labels, scores)

                ax2.set_xlabel("Similarity (%)")

                plt.tight_layout()

                st.pyplot(fig2)

                plt.close(fig2)

                # ----------------------------------------
                # Top Matches
                # ----------------------------------------

                st.subheader("🥇 Top Similar Papers")

                st.dataframe(
                    ranking_df.head(5),
                    use_container_width=True,
                    hide_index=True
                )

            else:

                st.info("No similarity rankings available.")

        else:

            st.info("👆 Click **Generate Similarity Matrix** to view the analysis.")
# =====================================================
# COMPARE PAPERS TAB
# =====================================================
# =====================================================
# AI PAPER COMPARISON
# =====================================================

with tab_compare:

    st.header("🤖 AI Research Paper Comparison")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    elif len(st.session_state.documents) < 2:

        st.warning("Upload at least two research papers.")

    else:

        paper_names = [

            doc["name"]

            for doc in st.session_state.documents

        ]

        col1, col2 = st.columns(2)

        with col1:

            paper1 = st.selectbox(

                "📄 Select Paper 1",

                paper_names,

                key="paper1"

            )

        with col2:

            remaining = [

                paper

                for paper in paper_names

                if paper != paper1

            ]

            paper2 = st.selectbox(

                "📄 Select Paper 2",

                remaining,

                key="paper2"

            )

        st.divider()

        if st.button(

            "🚀 Compare Papers",

            use_container_width=True

        ):

            doc1 = next(

                doc

                for doc in st.session_state.documents

                if doc["name"] == paper1

            )

            doc2 = next(

                doc

                for doc in st.session_state.documents

                if doc["name"] == paper2

            )

            context = f"""

Paper 1

Name:
{doc1["name"]}

Content:

{doc1["text"]}

------------------------------------------------------------

Paper 2

Name:
{doc2["name"]}

Content:

{doc2["text"]}

"""

            with st.spinner(

                "🤖 Gemini is comparing both papers..."

            ):

                try:

                    comparison = llm.compare_papers(

                        context

                    )

                except Exception as e:

                    comparison = str(e)

            st.success(

                "Comparison Completed Successfully!"

            )

            st.markdown(comparison)

            st.download_button(

                "⬇ Download Comparison Report",

                comparison,

                file_name="paper_comparison.md",

                mime="text/markdown"

            )

            st.divider()

            st.subheader("📊 Quick Statistics")

            col1, col2 = st.columns(2)

            col1.metric(

                "Paper 1 Words",

                len(doc1["text"].split())

            )

            col2.metric(

                "Paper 2 Words",

                len(doc2["text"].split())

            )

            col1.metric(

                "Paper 1 Pages",

                doc1["pages"]

                if isinstance(doc1["pages"], int)

                else len(doc1["pages"])

            )

            col2.metric(

                "Paper 2 Pages",

                doc2["pages"]

                if isinstance(doc2["pages"], int)

                else len(doc2["pages"])

            )

            st.divider()

            st.subheader("📑 Compared Papers")

            compare_df = pd.DataFrame({

                "Property": [

                    "Paper Name",

                    "Pages",

                    "Words"

                ],

                paper1: [

                    doc1["name"],

                    doc1["pages"]

                    if isinstance(doc1["pages"], int)

                    else len(doc1["pages"]),

                    len(doc1["text"].split())

                ],

                paper2: [

                    doc2["name"],

                    doc2["pages"]

                    if isinstance(doc2["pages"], int)

                    else len(doc2["pages"]),

                    len(doc2["text"].split())

                ]

            })

            st.dataframe(

                compare_df,

                use_container_width=True,

                hide_index=True

            )
# =====================================================
# KEYWORDS TAB
# =====================================================


with tab_keywords:

    st.header("🏷 AI Keyword Extraction")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button(
            "Extract Keywords",
            key="extract_keywords_tab_btn"
        ):

            context = ""

            for document in st.session_state.documents:

                context += f"""
Paper Name:
{document['name']}

Paper Content:
{document['text']}


--------------------------------------------------

"""

            with st.spinner("Extracting Keywords..."):

                try:

                    keywords = llm.extract_keywords(
                        context
                    )

                    st.session_state.keywords = keywords

                except Exception as e:

                    st.error(f"Error: {e}")

        # ------------------------------------------------
        # Display AI Keywords
        # ------------------------------------------------

        if st.session_state.keywords:

            st.success(
                "Keywords Extracted Successfully!"
            )

            st.subheader("🤖 AI Generated Keywords")

            st.markdown(
                st.session_state.keywords
            )
            if isinstance(st.session_state.keywords, dict):

                download_data = json.dumps(
                    st.session_state.keywords,
                    indent=4,
                    ensure_ascii=False
                )

                file_name = "keywords.json"
                mime = "application/json"

            else:

                download_data = str(st.session_state.keywords)

                file_name = "keywords.txt"
                mime = "text/plain"

            st.download_button(
                "⬇ Download AI Keywords",
                data=download_data,
                file_name=file_name,
                mime=mime,
                key="download_ai_keywords"
            )
           
        st.divider()

        # ====================================================
        # Keyword Frequency Analysis
        # ====================================================

        st.subheader("📊 Keyword Frequency Analysis")

        from collections import Counter
        from nltk.corpus import stopwords
        from nltk.tokenize import word_tokenize

        text = ""

        for document in st.session_state.documents:

            text += document["text"] + " "

        tokens = word_tokenize(
            text.lower()
        )

        stop_words = set(
            stopwords.words("english")
        )
        custom_stopwords = {

            "paper",
            "research",
            "results",
            "method",
            "methods",
            "model",
            "models",
            "using",
            "used",
            "proposed",
            "approach",
            "based",
            "study",
            "analysis",
            "figure",
            "table"

        }

        stop_words = stop_words.union(
            custom_stopwords
        )
        words = [

            word

            for word in tokens

            if word.isalpha()
            and word not in stop_words
            and len(word) > 2

        ]

        counter = Counter(words)

        top_keywords = counter.most_common(20)

        keyword_df = pd.DataFrame(

            top_keywords,

            columns=[
                "Keyword",
                "Frequency"
            ]

        )

        st.dataframe(

            keyword_df,

            use_container_width=True,

            hide_index=True

        )

        # ------------------------------------------------
        # Keyword Bar Chart
        # ------------------------------------------------

        fig, ax = plt.subplots(
            figsize=(10, 5)
        )

        ax.bar(

            keyword_df["Keyword"],

            keyword_df["Frequency"]

        )

        ax.set_title(
            "Top 20 Keywords"
        )

        plt.xticks(

            rotation=45,

            ha="right"

        )

        plt.tight_layout()

        st.pyplot(fig)

        # ------------------------------------------------
        # Statistics
        # ------------------------------------------------

        col1, col2 = st.columns(2)

        col1.metric(

            "Unique Keywords",

            len(counter)

        )

        col2.metric(

            "Top Keyword",

            top_keywords[0][0] if top_keywords else "N/A"

        )

        # ------------------------------------------------
        # Download CSV
        # ------------------------------------------------

        csv = keyword_df.to_csv(
            index=False
        )

        st.download_button(

            "⬇ Download Keyword Frequency",

            csv,

            file_name="keyword_frequency.csv",

            mime="text/csv"

        )

        st.success(

            f"Found {len(counter)} unique keywords across all uploaded papers."

        )           

# ====================================================
# ADVANCED WORD CLOUD ANALYTICS
# ====================================================

st.divider()

st.header("☁️ Research Word Cloud")
selected_paper = st.selectbox(

    "Generate Word Cloud For",

    ["All Papers"] +

    [

        doc["name"]

        for doc in st.session_state.documents

    ]

)

# ------------------------------------------------
# Settings
# ------------------------------------------------

col1, col2, col3 = st.columns(3)

with col1:
    bg = st.selectbox(
        "Background",
        ["white", "black"],
        key="wc_bg"
    )

with col2:
    color_theme = st.selectbox(
        "Color Theme",
        [
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis"
        ],
        key="wc_theme"
    )

with col3:
    max_words = st.slider(
        "Maximum Words",
        min_value=25,
        max_value=300,
        value=150,
        key="wc_max_words"
    )

stop_words = STOPWORDS
# --------------------------------------------
# Get text for Word Cloud
# --------------------------------------------

import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

# --------------------------------------------
# Get text for Word Cloud
# --------------------------------------------

documents = st.session_state.get("documents", [])

if not documents:

    st.warning("📄 Please upload a paper first.")

else:

    if selected_paper == "All Papers":

        text = " ".join(
            doc.get("text", "")
            for doc in documents
            if doc.get("text")
        )

    else:

        paper = next(
            (
                doc
                for doc in documents
                if doc.get("name") == selected_paper
            ),
            None,
        )

        text = paper.get("text", "") if paper else ""

    # --------------------------------------------
    # Generate Word Cloud
    # --------------------------------------------

    if text.strip():

        wordcloud = WordCloud(
            width=1800,
            height=900,
            background_color=bg,
            stopwords=STOPWORDS,
            max_words=max_words,
            colormap=color_theme,
            contour_width=2,
            contour_color="steelblue",
            prefer_horizontal=0.9,
            collocations=False,
            min_font_size=10,
            max_font_size=220,
            random_state=42,
            normalize_plurals=True,
            include_numbers=False,
        ).generate(text)

        import io

        fig_wc, ax_wc = plt.subplots(figsize=(15, 8))

        ax_wc.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax_wc.axis("off")

        st.pyplot(fig_wc)

# --------------------------------------------
# Download Word Cloud
# --------------------------------------------
        fig_wc, ax_wc = plt.subplots(figsize=(15, 8))

        ax_wc.imshow(
            wordcloud,
            interpolation="bilinear"
        )

        ax_wc.axis("off")

        st.pyplot(fig_wc)

        import io

        buffer = io.BytesIO()

        fig_wc.savefig(
            buffer,
            format="png",
            dpi=300,
            bbox_inches="tight"
        )

        buffer.seek(0)

        st.download_button(
            "⬇ Download Word Cloud",
            data=buffer,
            file_name="research_wordcloud.png",
            mime="image/png",
            key="download_wordcloud"
        )

        plt.close(fig_wc)

        # --------------------------------------------
        # Statistics
        # --------------------------------------------

        words = re.findall(
            r"[A-Za-z]+",
            text.lower()
        )

        words = [
            word
            for word in words
            if word not in STOPWORDS
        ]

        counter = Counter(words)

        st.subheader("📊 Word Cloud Statistics")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Words Analysed",
            f"{len(words):,}"
        )

        c2.metric(
            "Unique Words",
            f"{len(counter):,}"
        )

        c3.metric(
            "Displayed",
            max_words
        )

        top_word = counter.most_common(1)

        c4.metric(
            "Top Word",
            top_word[0][0] if top_word else "N/A"
        )

        if words:

            lexical_diversity = len(counter) / len(words)

            st.metric(
                "Lexical Diversity",
                f"{lexical_diversity:.3f}"
            )

        if counter:

            average_frequency = (
                sum(counter.values())
                / len(counter)
            )

            st.metric(
                "Average Frequency",
                f"{average_frequency:.2f}"
            )

    else:

        st.warning(
            "No text available to generate a Word Cloud."
        )

# ------------------------------------------------
# Top Keywords
# ------------------------------------------------

st.subheader("🔥 Top Keywords")

top_n = st.slider(
    "Number of Top Keywords",
    min_value=5,
    max_value=50,
    value=10,
    key="top_keywords_slider"
)

if documents and text.strip():

    words = re.findall(
        r"[A-Za-z]{2,}",
        text.lower()
    )

    words = [
        word
        for word in words
        if word not in STOPWORDS
    ]

    counter = Counter(words)

    top10 = pd.DataFrame(
        counter.most_common(top_n),
        columns=[
            "Keyword",
            "Frequency"
        ]
    )

    st.dataframe(
        top10,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("Upload papers to view top keywords.")


# ------------------------------------------------
# Pie Chart
# ------------------------------------------------

if documents and text.strip():

    st.subheader("🥧 Top Keyword Distribution")

    fig_pie, ax_pie = plt.subplots(figsize=(7, 7))

    ax_pie.pie(
        top10["Frequency"],
        labels=top10["Keyword"],
        autopct="%1.1f%%",
        startangle=90
    )

    ax_pie.axis("equal")

    st.pyplot(fig_pie)

    plt.close(fig_pie)

    # ------------------------------------------------
    # Word Length Distribution
    # ------------------------------------------------

    st.subheader("📏 Word Length Distribution")

    lengths = [
        len(word)
        for word in counter.keys()
    ]

    fig_len, ax_len = plt.subplots(figsize=(8, 4))

    ax_len.hist(
        lengths,
        bins=10
    )

    ax_len.set_xlabel("Word Length")
    ax_len.set_ylabel("Frequency")

    st.pyplot(fig_len)

    plt.close(fig_len)

else:

    st.info("Upload papers to view keyword charts.")

# ------------------------------------------------
# Rare Keywords + Search
# ------------------------------------------------

if documents and text.strip():

    st.subheader("🌟 Rare Keywords")

    rare_words = [

        word

        for word, freq in counter.items()

        if freq == 1

    ][:20]

    if rare_words:

        st.write(", ".join(rare_words))

    else:

        st.info("No rare keywords found.")

    # --------------------------------------------
    # Search Keyword
    # --------------------------------------------

    st.subheader("🔍 Search Keyword")

    search_word = st.text_input(
        "Enter keyword",
        key="search_keyword_input"
    )

    if search_word:

        frequency = counter.get(
            search_word.lower(),
            0
        )

        if frequency > 0:

            st.success(
                f'"{search_word}" appears {frequency} time(s).'
            )

        else:

            st.warning(
                "Keyword not found."
            )

else:

    st.info("📄 Upload papers to explore keywords.")
#-------------------------------------------
# Download Keyword Frequency Excel
# ------------------------------------------------

import io

# ------------------------------------------------
# Download Excel
# ------------------------------------------------

if not keyword_df.empty:
    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        keyword_df.to_excel(
            writer,
            index=False,
            sheet_name="Keywords"
        )

    output.seek(0)

    st.download_button(
        "⬇ Download Keywords Excel",
        data=output,
        file_name="keyword_frequency.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        key="download_keyword_excel"
    )

else:

    st.info("Generate keyword frequency first.")
# ------------------------------------------------
# Insights
# ------------------------------------------------

# ------------------------------------------------
# Word Cloud Insights
# ------------------------------------------------

if len(words) > 0:

    top_word = counter.most_common(1)

    st.info(f"""
### 📈 Word Cloud Insights

• Total Words Analysed: **{len(words):,}**

• Unique Keywords: **{len(counter):,}**

• Most Frequent Keyword: **{top_word[0][0] if top_word else "N/A"}**

• Frequency: **{top_word[0][1] if top_word else 0}**

• Displayed Words: **{max_words}**

• Theme: **{color_theme}**
""")

else:

    st.info("Generate a Word Cloud to view insights.")

# ------------------------------------------------
# Paper Statistics
# ------------------------------------------------

documents = st.session_state.get("documents", [])

paper_lengths = {
    doc["name"]: len(doc.get("text", "").split())
    for doc in documents
    if doc.get("text")
}

if paper_lengths:
    longest = max(paper_lengths, key=paper_lengths.get)
    shortest = min(paper_lengths, key=paper_lengths.get)

    col1, col2 = st.columns(2)
    col1.metric("📘 Longest Paper", longest)
    col2.metric("📗 Shortest Paper", shortest)
else:
    st.info("📂 No papers uploaded yet.")
# ------------------------------------------------
# Top Research Terms
# ------------------------------------------------

# ------------------------------------------------
# Top Research Terms
# ------------------------------------------------

st.subheader("🏆 Top Research Terms")
if len(counter) > 0:
    top5 = counter.most_common(5)

    for rank, (word, frequency) in enumerate(top5, start=1):
        st.write(f"**{rank}. {word}** — {frequency} occurrences")

else:
    st.info("Generate a Word Cloud first to view the top research terms.")
# Research Vocabulary
# ------------------------------------------------
# ------------------------------------------------
# Research Vocabulary
# ------------------------------------------------

st.subheader("📚 Research Vocabulary")

if "counter" in locals() and counter:

    longest_word = max(
        counter.keys(),
        key=len
    )

    shortest_word = min(
        counter.keys(),
        key=len
    )

    vocabulary_size = len(counter)

else:

    longest_word = "N/A"
    shortest_word = "N/A"
    vocabulary_size = 0

col1, col2, col3 = st.columns(3)

col1.metric(
    "Longest Keyword",
    longest_word
)

col2.metric(
    "Shortest Keyword",
    shortest_word
)

col3.metric(
    "Vocabulary Size",
    vocabulary_size
)

# =====================================================
# METADATA TAB
# =====================================================

with tab_metadata:

    st.header("📑 Research Paper Metadata")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Extract Metadata"):

            context = ""

            for document in st.session_state.documents:

                context += f"""

Paper Name:
{document['name']}

Paper Content:

{document['text']}

--------------------------------------------------

"""

            with st.spinner("Extracting Metadata..."):

                try:

                    metadata = llm.metadata(context)

                    # ----------------------------------------
                    # Save Metadata
                    # ----------------------------------------

                    st.session_state.metadata = metadata

                    # ----------------------------------------
                    # Convert Metadata to Dictionary
                    # ----------------------------------------

                    metadata_dict = {}

                    for line in metadata.split("\n"):

                        if ":" in line:

                            key, value = line.split(":", 1)

                            metadata_dict[key.strip()] = value.strip()

                    st.session_state.metadata_dict = metadata_dict

                    st.success(
                        "Metadata Extracted Successfully!"
                    )

                except Exception as e:

                    st.error(f"Error: {e}")

        # ----------------------------------------
        # Display Metadata
        # ----------------------------------------

        if st.session_state.metadata:

            st.subheader("Extracted Metadata")

            st.text_area(

                "Metadata",

                st.session_state.metadata,

                height=300

            )

            # ----------------------------------------
            # Metadata Table
            # ----------------------------------------

            metadata_df = pd.DataFrame(

                list(
                    st.session_state.metadata_dict.items()
                ),

                columns=[
                    "Field",
                    "Value"
                ]

            )

            st.subheader("📋 Metadata Table")

            st.dataframe(

                metadata_df,

                use_container_width=True,

                hide_index=True

            )

            # ----------------------------------------
            # Export CSV
            # ----------------------------------------

            csv = metadata_df.to_csv(
                index=False
            )

            st.download_button(

                "⬇ Export CSV",

                csv,

                file_name="paper_metadata.csv",

                mime="text/csv"

            )

            # ----------------------------------------
            # Export JSON
            # ----------------------------------------

            json_data = json.dumps(

                st.session_state.metadata_dict,

                indent=4

            )

            st.download_button(

                "⬇ Export JSON",

                json_data,

                file_name="paper_metadata.json",

                mime="application/json"

            )

            # ----------------------------------------
            # JSON Preview
            # ----------------------------------------

            st.subheader("📄 JSON Preview")

            st.code(

                json.dumps(

                    st.session_state.metadata_dict,

                    indent=4

                ),

                language="json"

            )

            # ----------------------------------------
            # Quick Metadata Summary
            # ----------------------------------------

            st.info(f"""
📄 **Title:** {st.session_state.metadata_dict.get("Title", "N/A")}

👥 **Authors:** {st.session_state.metadata_dict.get("Authors", "N/A")}

📅 **Year:** {st.session_state.metadata_dict.get("Year", "N/A")}

🏛 **Journal/Conference:** {st.session_state.metadata_dict.get("Journal/Conference", "N/A")}

🔗 **DOI:** {st.session_state.metadata_dict.get("DOI", "N/A")}

🔬 **Research Area:** {st.session_state.metadata_dict.get("Research Area", "N/A")}
""")

# =====================================================
# CITATION TAB
# =====================================================

with tab_citation:

    st.header("📚 Citation Generator")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        citation_style = st.selectbox(

            "Preferred Citation Style",

            [

                "All",

                "APA",

                "IEEE",

                "MLA",

                "BibTeX"

            ]

        )
        if st.session_state.metadata:

            with st.expander(
                "Extracted Metadata"
            ):

                st.text(
                    st.session_state.metadata
            )

        if st.button("Generate Citations"):

            context = ""

            for document in st.session_state.documents:

                context += f"""

Paper Name

{document['name']}

Paper Content

{document['text']}

------------------------------------

"""

            with st.spinner("Generating Citations..."):

                try:

                    citations = llm.citations(
                        context
                    )

                    st.session_state.citations = citations

                    st.success(
                        "Citations Generated Successfully!"
                    )

                    
                    st.text_area(

                        "Generated Citations",

                        citations,

                        height=350

                    )
                    st.download_button(

                        "⬇ Download Citations",

                        citations,

                        file_name="citations.txt",

                        mime="text/plain"

                    )

                except Exception as e:

                    st.error(e)

# =====================================================
# PAPER INFORMATION TAB
# =====================================================

# =====================================================
# PAPER INFORMATION TAB
# =====================================================

with tab_info:

    st.header("ℹ Research Paper Information")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        # ------------------------------------------------
        # Overall Statistics
        # ------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📄 Total Papers",
            len(st.session_state.documents)
        )

        col2.metric(
            "📚 Total Pages",
            sum(
                len(doc["pages"])
                if isinstance(doc["pages"], list)
                else doc["pages"]
                for doc in st.session_state.documents
            )
        )

        col3.metric(
            "📝 Total Words",
            sum(
                len(doc["text"].split())
                for doc in st.session_state.documents
            )
        )

        st.divider()

        # ------------------------------------------------
        # Paper Selector
        # ------------------------------------------------

        selected_paper = st.selectbox(

            "📄 Select Research Paper",

            [doc["name"] for doc in st.session_state.documents]

        )

        document = next(

            doc

            for doc in st.session_state.documents

            if doc["name"] == selected_paper

        )

        # ------------------------------------------------
        # Paper Statistics
        # ------------------------------------------------

        words = len(
            document["text"].split()
        )

        characters = len(
            document["text"]
        )

        pages = (
            len(document["pages"])
            if isinstance(document["pages"], list)
            else document["pages"]
        )

        paragraphs = len(

            [
                p

                for p in document["text"].split("\n\n")

                if p.strip()
            ]

        )

        reading_time = round(

            words / 200,

            1

        )

        average_words_per_page = round(
            words / pages,
            2
        ) if pages else 0

        average_chars_per_word = round(
            characters / words,
            2
        ) if words else 0

        chunk_count = len(

            [

                chunk

                for chunk in st.session_state.chunks

                if chunk["source"] == document["name"]

            ]

        )

        # ------------------------------------------------
        # Metrics
        # ------------------------------------------------

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "📚 Pages",
            pages
        )

        col2.metric(
            "🧩 Chunks",
            chunk_count
        )

        col3.metric(
            "⏱ Reading Time",
            f"{reading_time} min"
        )

        col4, col5, col6, col7, col8 = st.columns(5)

        col4.metric(
            "📝 Words",
            words
        )

        col5.metric(
            "🔤 Characters",
            characters
        )

        col6.metric(
            "📄 Paragraphs",
            paragraphs
        )
        col7.metric(
            "📖 Avg Words/Page",
            average_words_per_page
        )

        col8.metric(
            "🔠 Avg Chars/Word",
            average_chars_per_word
        )

        st.divider()
        st.subheader("📈 Reading Statistics")

        st.write("Reading Progress")

        st.progress(min(words / 10000, 1.0))

        st.caption(
        f"{words:,} words"
        )   

        # ------------------------------------------------
        # Paper Preview
        # ------------------------------------------------

        st.subheader("📖 Paper Preview")

        st.text_area(

            "First 1000 Characters",

            value=document["text"][:1000],

            height=300,

            key=f"preview_{document['name']}"

        )
        with st.expander(
            "📄 View Complete Paper"
        ):

            st.text(document["text"])
        # ------------------------------------------------
        # Download Paper
        # ------------------------------------------------

        st.download_button(

            "⬇ Download Paper Text",

            data=document["text"],

            file_name=f"{document['name']}.txt",

            mime="text/plain",

            key=f"download_{document['name']}"

        )

        st.divider()

        # ------------------------------------------------
        # File Information
        # ------------------------------------------------

        st.subheader("📂 File Information")

        info_df = pd.DataFrame({

            "Property": [

                "Paper Name",

                "Pages",

                "Words",

                "Characters",

                "Paragraphs",

                "Chunks",

                "Estimated Reading Time"

            ],

            "Value": [

                document["name"],

                pages,

                words,

                characters,

                paragraphs,

                chunk_count,

                f"{reading_time} minutes"

            ]

        })

        st.dataframe(

            info_df,

            use_container_width=True,

            hide_index=True

        )

        summary = f"""
        Paper Name : {document['name']}

        Pages : {pages}

        Words : {words}

        Characters : {characters}

        Paragraphs : {paragraphs}

        Chunks : {chunk_count}

        Reading Time : {reading_time} min
        """

        st.download_button(

            "⬇ Download Paper Statistics",

                summary,

                file_name="paper_statistics.txt",

                 mime="text/plain"
            )
# =====================================================
# EXTRACTED METADATA
# =====================================================

if st.session_state.get("metadata"):

    st.divider()

    st.subheader("📑 Extracted Metadata")

    with st.expander(
        "View Metadata",
        expanded=False
    ):

        st.text(
            st.session_state.metadata
        )

# =====================================================
# RESEARCH ANALYTICS DASHBOARD
# =====================================================

st.divider()
st.header("📊 Research Analytics Dashboard")

analytics = []

# -----------------------------------------------------
# Build Analytics
# -----------------------------------------------------

for doc in st.session_state.documents:

    page_count = (
        len(doc["pages"])
        if isinstance(doc["pages"], list)
        else doc["pages"]
    )

    word_count = len(doc["text"].split())

    analytics.append({
        "Paper": doc["name"],
        "Pages": page_count,
        "Words": word_count,
        "Average Words/Page": (
            round(word_count / page_count, 2)
            if page_count > 0
            else 0
        )
    })

analytics_df = pd.DataFrame(
    analytics,
    columns=[
        "Paper",
        "Pages",
        "Words",
        "Average Words/Page"
    ]
)

# -----------------------------------------------------
# No Papers Uploaded
# -----------------------------------------------------

if analytics_df.empty:

    st.info("📄 Upload one or more research papers to view analytics.")

else:

    # =====================================================
    # SUMMARY METRICS
    # =====================================================

    st.subheader("📈 Overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Papers",
        len(analytics_df)
    )

    col2.metric(
        "Total Pages",
        int(analytics_df["Pages"].sum())
    )

    col3.metric(
        "Total Words",
        int(analytics_df["Words"].sum())
    )

    col4.metric(
        "Avg Words/Page",
        round(
            analytics_df["Average Words/Page"].mean(),
            2
        )
    )

    st.divider()

    # =====================================================
    # ANALYTICS TABLE
    # =====================================================

    st.subheader("📋 Paper Analytics")

    st.dataframe(
        analytics_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================================
    # WORDS PER PAPER
    # =====================================================

    st.subheader("📊 Words Per Paper")

    fig1, ax1 = plt.subplots(figsize=(10,5))

    ax1.bar(
        analytics_df["Paper"],
        analytics_df["Words"]
    )

    ax1.set_ylabel("Words")
    ax1.set_xlabel("Paper")
    ax1.set_title("Words Distribution")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig1)

    # =====================================================
    # PAGES PER PAPER
    # =====================================================

    st.subheader("📄 Pages Per Paper")

    fig2, ax2 = plt.subplots(figsize=(10,5))

    ax2.bar(
        analytics_df["Paper"],
        analytics_df["Pages"]
    )

    ax2.set_ylabel("Pages")
    ax2.set_xlabel("Paper")
    ax2.set_title("Pages Distribution")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig2)

    # =====================================================
    # WORD DENSITY
    # =====================================================

    st.subheader("📚 Average Words per Page")

    fig3, ax3 = plt.subplots(figsize=(10,5))

    ax3.bar(
        analytics_df["Paper"],
        analytics_df["Average Words/Page"]
    )

    ax3.set_ylabel("Average Words/Page")
    ax3.set_xlabel("Paper")
    ax3.set_title("Reading Density")

    plt.xticks(rotation=30)
    plt.tight_layout()

    st.pyplot(fig3)

    st.divider()

    # =====================================================
    # LARGEST & SMALLEST PAPER
    # =====================================================

    largest = analytics_df.loc[
        analytics_df["Words"].idxmax()
    ]

    smallest = analytics_df.loc[
        analytics_df["Words"].idxmin()
    ]

    longest_pages = analytics_df.loc[
        analytics_df["Pages"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "📘 Largest Paper",
        largest["Paper"],
        f'{largest["Words"]:,} words'
    )

    col2.metric(
        "📗 Smallest Paper",
        smallest["Paper"],
        f'{smallest["Words"]:,} words'
    )

    col3.metric(
        "📄 Most Pages",
        longest_pages["Paper"],
        f'{longest_pages["Pages"]} pages'
    )

    st.divider()

    # =====================================================
    # QUICK INSIGHTS
    # =====================================================

    st.subheader("💡 Research Insights")

    st.success(
        f"""
• Total Papers: **{len(analytics_df)}**

• Total Pages: **{int(analytics_df['Pages'].sum())}**

• Total Words: **{int(analytics_df['Words'].sum()):,}**

• Largest Paper: **{largest['Paper']}**

• Smallest Paper: **{smallest['Paper']}**

• Average Words per Page: **{analytics_df['Average Words/Page'].mean():.2f}**
"""
    )
# =====================================================
# DOWNLOAD ANALYTICS
# =====================================================

st.download_button(

    "⬇ Download Analytics CSV",

    analytics_df.to_csv(index=False),

    file_name="research_analytics.csv",

    mime="text/csv"

)

# =====================================================
# RESEARCH REPORT GENERATOR
# =====================================================

st.divider()

st.header("📑 Research Report Generator")

if st.button("📝 Generate Research Report"):

    report = f"""
========================================================
            AI RESEARCH ASSISTANT REPORT
========================================================

Report Generated Successfully

========================================================
OVERALL SUMMARY
========================================================

Total Papers : {len(st.session_state.documents)}

Total Pages : {analytics_df['Pages'].sum()}

Total Words : {analytics_df['Words'].sum()}

Average Words/Page :
{round(analytics_df['Words'].sum() / analytics_df['Pages'].sum(), 2)}

========================================================
PAPER ANALYTICS
========================================================

"""

    # ------------------------------------------------
    # Individual Paper Statistics
    # ------------------------------------------------

    for _, row in analytics_df.iterrows():

        report += f"""
Paper Name
----------
{row['Paper']}

Pages
{row['Pages']}

Words
{row['Words']}

Average Words/Page
{row['Average Words/Page']}

--------------------------------------------------------

"""

    # ------------------------------------------------
    # Metadata
    # ------------------------------------------------

    if st.session_state.get("metadata"):

        report += """
========================================================
EXTRACTED METADATA
========================================================

"""

        report += st.session_state.metadata

        report += "\n\n"

    # ------------------------------------------------
    # Top Keywords
    # ------------------------------------------------

    
    if len(counter) > 0:

        report += """
========================================================
TOP RESEARCH KEYWORDS
========================================================

"""

        for word, freq in counter.most_common(20):

            report += f"{word:<25} {freq}\n"

        report += "\n"

    # ------------------------------------------------
    # Research Vocabulary
    # ------------------------------------------------

    if len(counter) > 0:

        report += """
========================================================
VOCABULARY STATISTICS
========================================================

"""

        report += f"""

Unique Keywords :
{len(counter)}

Most Frequent Keyword :
{counter.most_common(1)[0][0]}

Frequency :
{counter.most_common(1)[0][1]}

Longest Keyword :
{max(counter.keys(), key=len)}

Shortest Keyword :
{min(counter.keys(), key=len)}

"""

    # ------------------------------------------------
    # Reading Statistics
    # ------------------------------------------------

    report += """
========================================================
READING STATISTICS
========================================================

"""

    total_words = analytics_df["Words"].sum()

    report += f"""

Estimated Reading Time :

{round(total_words / 200, 1)} minutes

"""

    # ------------------------------------------------
    # Largest / Smallest Paper
    # ------------------------------------------------

    largest = analytics_df.loc[
        analytics_df["Words"].idxmax()
    ]

    smallest = analytics_df.loc[
        analytics_df["Words"].idxmin()
    ]

    report += f"""
========================================================
PAPER COMPARISON
========================================================

Largest Paper

{largest['Paper']}

Words

{largest['Words']}

----------------------------------------

Smallest Paper

{smallest['Paper']}

Words

{smallest['Words']}

"""

    # ------------------------------------------------
    # Report Preview
    # ------------------------------------------------

    st.subheader("📖 Report Preview")

    st.text_area(

        "Generated Report",

        report,

        height=450

    )

    # ------------------------------------------------
    # Report Statistics
    # ------------------------------------------------

    st.subheader("📊 Report Statistics")

    col1, col2, col3 = st.columns(3)

    col1.metric(

        "Lines",

        len(report.splitlines())

    )

    col2.metric(

        "Words",

        len(report.split())

    )

    col3.metric(

        "Characters",

        len(report)

    )

    # ------------------------------------------------
    # Downloads
    # ------------------------------------------------

    st.download_button(

        "⬇ Download TXT Report",

        report,

        file_name="research_report.txt",

        mime="text/plain"

    )

    markdown_report = report.replace(

        "========================================================",

        "---"

    )

    st.download_button(

        "⬇ Download Markdown Report",

        markdown_report,

        file_name="research_report.md",

        mime="text/markdown"

    )

    # ------------------------------------------------
    # Success Message
    # ------------------------------------------------

    st.success(

        "✅ Research Report Generated Successfully!"

    )

# =====================================================
# SESSION SUMMARY DASHBOARD
# =====================================================

st.divider()

st.header("📊 Session Summary Dashboard")

# ------------------------------------------------
# Session Statistics
# ------------------------------------------------

total_papers = len(st.session_state.documents)

total_pages = analytics_df["Pages"].sum()

total_words = analytics_df["Words"].sum()

total_chunks = len(st.session_state.chunks)

reading_time = round(
    total_words / 200,
    1
)

# ------------------------------------------------
# Top Metrics
# ------------------------------------------------

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "📄 Papers",
    total_papers
)

col2.metric(
    "📚 Pages",
    total_pages
)

col3.metric(
    "📝 Words",
    f"{total_words:,}"
)

col4.metric(
    "🧩 Chunks",
    total_chunks
)

col5, col6 = st.columns(2)

col5.metric(
    "⏱ Reading Time",
    f"{reading_time} min"
)

col6.metric(
    "📖 Avg Words/Page",
    round(
        total_words / total_pages,
        2
    ) if total_pages else 0
)

st.divider()

# =====================================================
# AI Usage Statistics
# =====================================================

st.subheader("🤖 AI Usage Statistics")

user_questions = sum(

    1

    for msg in st.session_state.messages

    if msg["role"] == "user"

)

assistant_answers = sum(

    1

    for msg in st.session_state.messages

    if msg["role"] == "assistant"

)

col1, col2 = st.columns(2)

col1.metric(
    "❓ Questions Asked",
    user_questions
)

col2.metric(
    "🤖 AI Responses",
    assistant_answers
)

# =====================================================
# Database Status
# =====================================================

st.subheader("💾 Database Status")

db_col1, db_col2 = st.columns(2)

db_col1.metric(
    "Stored Chunks",
    st.session_state.database.count()
)

db_col2.metric(
    "Database Ready",
    "✅ Yes" if st.session_state.database_ready else "❌ No"
)

# =====================================================
# Research Quality Score
# =====================================================

st.subheader("⭐ Research Quality Score")

score = 0

if total_papers >= 3:
    score += 25

if total_pages >= 30:
    score += 25

if total_words >= 10000:
    score += 20

if total_chunks >= 50:
    score += 20

if assistant_answers >= 5:
    score += 10

st.progress(score / 100)

if score >= 90:

    grade = "Excellent"

elif score >= 75:

    grade = "Very Good"

elif score >= 60:

    grade = "Good"

else:

    grade = "Needs Improvement"

st.success(
    f"Overall Score: {score}/100 ({grade})"
)

# =====================================================
# Research Health
# =====================================================

st.subheader("🩺 Research Health Indicators")

health = pd.DataFrame({

    "Indicator":[

        "Uploaded ≥ 3 Papers",

        "Pages ≥ 30",

        "Words ≥ 10,000",

        "Chunks ≥ 50",

        "Database Ready",

        "AI Chat Used"

    ],

    "Status":[

        "✅" if total_papers >= 3 else "❌",

        "✅" if total_pages >= 30 else "❌",

        "✅" if total_words >= 10000 else "❌",

        "✅" if total_chunks >= 50 else "❌",

        "✅" if st.session_state.database_ready else "❌",

        "✅" if assistant_answers > 0 else "❌"

    ]

})

st.dataframe(

    health,

    use_container_width=True,

    hide_index=True

)

# =====================================================
# Largest & Smallest Paper
# =====================================================
# =====================================================
# Paper Insights
# =====================================================

st.subheader("📚 Paper Insights")

if analytics_df.empty or "Words" not in analytics_df.columns:

    st.info("📂 Upload one or more research papers to view insights.")

    largest = {
        "Paper": "N/A",
        "Pages": 0,
        "Words": 0,
        "Average Words/Page": 0,
    }

    smallest = largest.copy()

else:

    largest = analytics_df.loc[
        analytics_df["Words"].idxmax()
    ]

    smallest = analytics_df.loc[
        analytics_df["Words"].idxmin()
    ]

    col1, col2 = st.columns(2)

    with col1:

        st.success(f"""
### 📘 Largest Paper

**{largest['Paper']}**

Pages: {largest['Pages']}

Words: {largest['Words']}

Average Words/Page: {largest['Average Words/Page']}
""")

    with col2:

        st.info(f"""
### 📗 Smallest Paper

**{smallest['Paper']}**

Pages: {smallest['Pages']}

Words: {smallest['Words']}

Average Words/Page: {smallest['Average Words/Page']}
""")

# =====================================================
# Quick Insights
# =====================================================

st.subheader("💡 Quick Insights")

insights = [

    f"📄 Total research papers analysed: {total_papers}",

    f"📚 Total pages processed: {total_pages}",

    f"📝 Total words analysed: {total_words:,}",

    f"🧩 Text chunks indexed: {total_chunks}",

    f"🤖 AI responses generated: {assistant_answers}",

    f"⭐ Overall research score: {score}/100"

]

for item in insights:
    st.write("•", item)

# =====================================================
# Download Session Summary
# =====================================================

summary = f"""
AI RESEARCH ASSISTANT SESSION SUMMARY

----------------------------------------

Total Papers : {total_papers}

Total Pages : {total_pages}

Total Words : {total_words}

Total Chunks : {total_chunks}

Estimated Reading Time : {reading_time} minutes

Questions Asked : {user_questions}

AI Responses : {assistant_answers}

Research Quality Score : {score}/100

Database Ready : {st.session_state.database_ready}

Largest Paper : {largest['Paper']}

Smallest Paper : {smallest['Paper']}

----------------------------------------
"""

st.download_button(

    "⬇ Download Session Summary",

    summary,

    file_name="session_summary.txt",

    mime="text/plain"

)

st.success("✅ Session Summary Generated Successfully!")

# =====================================================
# RESEARCH INSIGHTS DASHBOARD
# =====================================================

st.divider()

st.header("🔬 Research Insights Dashboard")

if analytics_df.empty:

    st.warning("📂 Upload research papers to generate the dashboard.")

else:

    total_papers = len(st.session_state.documents)

    total_pages = analytics_df["Pages"].sum()

    total_words = analytics_df["Words"].sum()

    avg_words = round(
        total_words / total_papers,
        2
    ) if total_papers else 0

    avg_pages = round(
        total_pages / total_papers,
        2
    ) if total_papers else 0

    largest = analytics_df.loc[
        analytics_df["Words"].idxmax()
    ]

    smallest = analytics_df.loc[
        analytics_df["Words"].idxmin()
    ]

    # ------------------------------------------------
    # Metrics
    # ------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📄 Avg Words / Paper",
        f"{avg_words:,.0f}"
    )

    col2.metric(
        "📚 Avg Pages / Paper",
        avg_pages
    )

    col3.metric(
        "📘 Largest Paper",
        f"{largest['Words']:,}"
    )

    col4.metric(
        "📗 Smallest Paper",
        f"{smallest['Words']:,}"
    )

    # ------------------------------------------------
    # Research Distribution
    # ------------------------------------------------

    st.subheader("📊 Research Distribution")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.hist(
        analytics_df["Words"],
        bins=max(1, min(10, len(analytics_df)))
    )

    ax.set_xlabel("Words")
    ax.set_ylabel("Number of Papers")

    st.pyplot(fig)

# ------------------------------------------------
# Average Paper Size
# ------------------------------------------------

# ------------------------------------------------
# Average Paper Size
# ------------------------------------------------

st.subheader("📈 Average Paper Size")

if not analytics_df.empty:

    avg_words = analytics_df["Words"].mean()

    fig2, ax2 = plt.subplots(figsize=(8, 4))

    ax2.plot(
        analytics_df["Words"],
        marker="o",
        linewidth=2,
        label="Paper Size"
    )

    ax2.axhline(
        y=avg_words,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Average ({avg_words:.0f} words)"
    )

    ax2.set_xlabel("Paper")
    ax2.set_ylabel("Words")
    ax2.set_xticks(range(len(analytics_df)))
    ax2.set_xticklabels(
        analytics_df["Paper"],
        rotation=45,
        ha="right"
    )

    ax2.legend()

    st.pyplot(fig2)

else:
    st.info("No papers uploaded yet.")
# ------------------------------------------------
# Top Paper
# ------------------------------------------------

st.subheader("🏆 Top Research Paper")

st.success(f"""

**Paper**

{largest['Paper']}

**Words**

{largest['Words']}

**Pages**

{largest['Pages']}

Average Words/Page

{largest['Average Words/Page']}

""")

# ------------------------------------------------
# Quick Insights
# ------------------------------------------------

st.subheader("💡 AI Insights")

insights = []

if total_papers >= 5:

    insights.append(
        "✅ Good number of research papers uploaded."
    )

else:

    insights.append(
        "⚠ Upload more papers for stronger analysis."
    )

if total_pages >= 50:

    insights.append(
        "✅ Large research collection detected."
    )

if total_words >= 30000:

    insights.append(
        "✅ Excellent dataset for semantic search."
    )

if total_chunks >= 100:

    insights.append(
        "✅ Vector database has rich coverage."
    )

for insight in insights:

    st.info(insight)

# ------------------------------------------------
# Download Insights
# ------------------------------------------------

insight_report = "\n".join(insights)

st.download_button(

    "⬇ Download Insights",

    insight_report,

    file_name="research_insights.txt",

    mime="text/plain"

)
# =====================================================
# AI RESEARCH READINESS DASHBOARD
# =====================================================

st.divider()

st.header("🚀 AI Research Readiness")

# ------------------------------------------------
# Calculate Scores
# ------------------------------------------------

score = 0

checks = {

    "Minimum 3 Papers":
        len(st.session_state.documents) >= 3,

    "Minimum 30 Pages":
        analytics_df["Pages"].sum() >= 30,

    "Minimum 10,000 Words":
        analytics_df["Words"].sum() >= 10000,

    "Minimum 50 Chunks":
        len(st.session_state.chunks) >= 50,

    "Vector Database Ready":
        st.session_state.database_ready,

    "AI Chat Used":
        len(st.session_state.messages) > 0

}

score = sum(checks.values()) / len(checks) * 100

# ------------------------------------------------
# Overall Readiness
# ------------------------------------------------

st.subheader("📊 Overall Readiness")

st.progress(score / 100)

if score >= 90:

    grade = "🟢 Excellent"

elif score >= 75:

    grade = "🟡 Good"

elif score >= 50:

    grade = "🟠 Moderate"

else:

    grade = "🔴 Poor"

st.metric(

    "Readiness Score",

    f"{score:.0f}%"

)

st.success(f"Overall Grade: {grade}")

# ------------------------------------------------
# Checklist
# ------------------------------------------------

st.subheader("✅ Readiness Checklist")

status_df = pd.DataFrame({

    "Requirement": list(checks.keys()),

    "Status": [

        "✅ Passed"

        if value else

        "❌ Failed"

        for value in checks.values()

    ]

})

st.dataframe(

    status_df,

    hide_index=True,

    use_container_width=True

)

# ------------------------------------------------
# Recommendations
# ------------------------------------------------

st.subheader("💡 Recommendations")

recommendations = []

if len(st.session_state.documents) < 3:

    recommendations.append(
        "Upload more research papers."
    )

if analytics_df["Pages"].sum() < 30:

    recommendations.append(
        "Increase the number of pages for richer analysis."
    )

if analytics_df["Words"].sum() < 10000:

    recommendations.append(
        "Upload longer papers."
    )

if len(st.session_state.chunks) < 50:

    recommendations.append(
        "Generate more text chunks."
    )

if not st.session_state.database_ready:

    recommendations.append(
        "Create embeddings and store them in the vector database."
    )

if len(st.session_state.messages) == 0:

    recommendations.append(
        "Use the AI Chat tab to interact with your papers."
    )

if recommendations:

    for item in recommendations:

        st.warning(item)

else:

    st.success(
        "🎉 Your research collection is ready for advanced AI analysis."
    )

# ------------------------------------------------
# Readiness Summary
# ------------------------------------------------

st.subheader("📋 Readiness Summary")

summary = f"""
AI RESEARCH READINESS REPORT

Readiness Score : {score:.0f}%

Grade : {grade}

Total Papers : {len(st.session_state.documents)}

Total Pages : {analytics_df['Pages'].sum()}

Total Words : {analytics_df['Words'].sum()}

Total Chunks : {len(st.session_state.chunks)}

Database Ready : {st.session_state.database_ready}
"""

st.text_area(

    "Summary",

    summary,

    height=220

)

st.download_button(

    "⬇ Download Readiness Report",

    summary,

    file_name="research_readiness.txt",

    mime="text/plain"

)


# =====================================================
# SEARCH PAPERS TAB
# =====================================================

with tab_search:

    st.header("🔍 Search Research Papers")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        search_text = st.text_input(
            "Enter a keyword, phrase, or paper name"
        )

        if search_text:

            matches = []

            for document in st.session_state.documents:

                paper_name = document["name"].lower()

                paper_text = document["text"].lower()

                if (
                    search_text.lower() in paper_name
                    or
                    search_text.lower() in paper_text
                ):

                    matches.append(document)

            # ----------------------------------------
            # Search Summary
            # ----------------------------------------

            st.success(
                f"Found {len(matches)} matching paper(s)."
            )

            col1, col2 = st.columns(2)

            col1.metric(
                "Matching Papers",
                len(matches)
            )

            total_words = sum(
                len(doc["text"].split())
                for doc in matches
            )

            col2.metric(
                "Words Covered",
                total_words
            )

            st.divider()

            # ----------------------------------------
            # No Match
            # ----------------------------------------

            if len(matches) == 0:

                st.warning(
                    "No papers matched your search."
                )

            # ----------------------------------------
            # Display Matches
            # ----------------------------------------

            else:

                report = ""

                for index, document in enumerate(matches, start=1):

                    occurrences = document["text"].lower().count(
                        search_text.lower()
                    )

                    pages = (
                        len(document["pages"])
                        if isinstance(document["pages"], list)
                        else document["pages"]
                    )

                    words = len(
                        document["text"].split()
                    )

                    with st.expander(
                        f"📄 {document['name']}",
                        expanded=False
                    ):

                        col1, col2, col3 = st.columns(3)

                        col1.metric(
                            "Pages",
                            pages
                        )

                        col2.metric(
                            "Words",
                            words
                        )

                        col3.metric(
                            "Occurrences",
                            occurrences
                        )

                        position = document["text"].lower().find(
                            search_text.lower()
                        )

                        if position == -1:

                            preview = document["text"][:500]

                        else:

                            start = max(
                                0,
                                position - 250
                            )

                            end = min(
                                len(document["text"]),
                                position + 500
                            )

                            preview = document["text"][start:end]

                        st.subheader("Matching Preview")

                        st.text_area(

                            "Preview",

                            preview,

                            height=250,

                            key=f"preview_{index}"

                        )

                        st.download_button(

                            "⬇ Download Paper",

                            document["text"],

                            file_name=f"{document['name']}.txt",

                            mime="text/plain",

                            key=f"download_{index}"

                        )

                    report += f"""
Paper : {document['name']}

Pages : {pages}

Words : {words}

Occurrences : {occurrences}

------------------------------------------------------

"""

                # ----------------------------------------
                # Download Search Report
                # ----------------------------------------

                st.divider()

                st.download_button(

                    "⬇ Download Search Report",

                    report,

                    file_name="search_report.txt",

                    mime="text/plain"

                )

# =====================================================
# STATISTICS TAB
# =====================================================

with tab_stats:

    st.header("📊 Analytics Dashboard")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        stats = st.session_state.statistics

        # ------------------------------------------------
        # Top Metrics
        # ------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "📄 Papers",
            stats.get("papers", 0)
        )

        col2.metric(
            "📚 Pages",
            stats.get("pages", 0)
        )

        col3.metric(
            "🧩 Chunks",
            stats.get("chunks", 0)
        )

        col4.metric(
            "📝 Words",
            stats.get("words", 0)
        )

        st.divider()

        # ------------------------------------------------
        # Statistics Data
        # ------------------------------------------------

        chart_df = pd.DataFrame({

            "Metric": [
                "Words",
                "Characters",
                "Chunks",
                "Pages",
                "Paragraphs"
            ],

            "Value": [
                stats.get("words", 0),
                stats.get("characters", 0),
                stats.get("chunks", 0),
                stats.get("pages", 0),
                stats.get("paragraphs", 0)
            ]

        })

        # ------------------------------------------------
        # Bar Chart
        # ------------------------------------------------

        st.subheader("📈 Overall Statistics")

        fig = px.bar(

            chart_df,

            x="Metric",

            y="Value",

            text="Value",

            title="Research Paper Statistics"

        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # ------------------------------------------------
        # Pie Chart
        # ------------------------------------------------

        st.subheader("🥧 Distribution")

        if chart_df["Value"].sum() > 0:

            pie = px.pie(

                chart_df,

                values="Value",

                names="Metric",

                hole=0.4

            )

            st.plotly_chart(

                pie,

                use_container_width=True

            )

        else:

            st.info("No data available for Pie Chart.")

        # ------------------------------------------------
        # Uploaded Papers
        # ------------------------------------------------

        st.subheader("📄 Uploaded Papers")

        papers = []

        for doc in st.session_state.documents:

            pages = doc["pages"]

            if isinstance(pages, int):
                page_count = pages
            else:
                page_count = len(pages)

            papers.append({

                "Paper": doc["name"],

                "Pages": page_count,

                "Words": len(doc["text"].split()),

                "Characters": len(doc["text"])

            })

        paper_df = pd.DataFrame(papers)

        st.dataframe(

            paper_df,

            use_container_width=True,

            hide_index=True

        )

        # ------------------------------------------------
        # Paper Comparison
        # ------------------------------------------------

        if not paper_df.empty:

            st.subheader("📚 Paper Length Comparison")

            fig = px.bar(

                paper_df,

                x="Paper",

                y="Words",

                color="Pages",

                text="Words"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        # ------------------------------------------------
        # Chunk Distribution
        # ------------------------------------------------

        if st.session_state.chunks:

            st.subheader("🧩 Chunks Per Paper")

            chunk_df = pd.DataFrame([

                {

                    "Paper": chunk["source"]

                }

                for chunk in st.session_state.chunks

            ])

            chunk_count = (

                chunk_df

                .groupby("Paper")

                .size()

                .reset_index(name="Chunks")

            )

            fig = px.bar(

                chunk_count,

                x="Paper",

                y="Chunks",

                text="Chunks"

            )

            st.plotly_chart(

                fig,

                use_container_width=True

            )

        # ------------------------------------------------
        # Treemap
        # ------------------------------------------------

        if not paper_df.empty:

            st.subheader("🌳 Word Distribution")

            tree = px.treemap(

                paper_df,

                path=["Paper"],

                values="Words"

            )

            st.plotly_chart(

                tree,

                use_container_width=True

            )

        # ------------------------------------------------
        # Radar Chart
        # ------------------------------------------------

        st.subheader("🕸 Research Metrics")

        radar = go.Figure()

        radar.add_trace(

            go.Scatterpolar(

                r=[

                    stats.get("pages", 0),

                    stats.get("chunks", 0),

                    stats.get("words", 0),

                    stats.get("paragraphs", 0)

                ],

                theta=[

                    "Pages",

                    "Chunks",

                    "Words",

                    "Paragraphs"

                ],

                fill="toself"

            )

        )

        radar.update_layout(

            polar=dict(

                radialaxis=dict(

                    visible=True

                )

            ),

            showlegend=False

        )

        st.plotly_chart(

            radar,

            use_container_width=True

        )

        # ------------------------------------------------
        # Database Information
        # ------------------------------------------------

        st.subheader("🗄 Database Information")

        database_df = pd.DataFrame({

            "Property": [

                "Embedding Model",

                "Embedding Dimension",

                "Vector Database",

                "Stored Chunks"

            ],

            "Value": [

                "all-MiniLM-L6-v2",

                stats.get("embedding_dimension", 384),

                "ChromaDB",

                st.session_state.database.count()

            ]

        })

        st.table(database_df)

# =====================================================
# KEYWORDS TAB
# =====================================================

with tab_keywords:

    st.header("🏷 Research Keywords")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        if st.button("Extract Keywords"):

            context = ""

            for document in st.session_state.documents:

                context += document["text"]

                context += "\n\n"

            with st.spinner("Extracting Keywords..."):

                try:

                    keywords = llm.keywords(context)

                    st.success("Keywords Extracted")

                    st.markdown(keywords)

                    st.download_button(

                        "⬇ Download Keywords",

                        keywords,

                        file_name="keywords.txt",

                        mime="text/plain"

                    )

                except Exception as e:

                    st.error(e)
                    
# =====================================================
# RESEARCH TREND ANALYSIS
# =====================================================

with tab_trends:

    st.header("📈 Research Trend Analysis")

    if not st.session_state.database_ready:

        st.info(
            "Please upload research papers first."
        )

    else:

        trends = trend.dataframe(

            st.session_state.documents,

            top_n=30

        )

        # ------------------------------------------
        # Metrics
        # ------------------------------------------

        total_keywords = trends["Frequency"].sum()

        unique_keywords = len(trends)

        top_keyword = trends.iloc[0]["Keyword"]

        top_frequency = trends.iloc[0]["Frequency"]

        col1, col2, col3 = st.columns(3)

        col1.metric(

            "🔑 Unique Keywords",

            unique_keywords

        )

        col2.metric(

            "🔥 Top Keyword",

            top_keyword

        )

        col3.metric(

            "📊 Frequency",

            top_frequency

        )

        st.divider()

        # ------------------------------------------
        # Keyword Table
        # ------------------------------------------

        st.subheader("📋 Top Research Keywords")

        st.dataframe(

            trends,

            use_container_width=True,

            hide_index=True

        )

        # ------------------------------------------
        # Keyword Frequency Chart
        # ------------------------------------------

        st.subheader("📊 Keyword Frequency")

        fig, ax = plt.subplots(

            figsize=(12,5)

        )

        ax.bar(

            trends["Keyword"],

            trends["Frequency"]

        )

        ax.set_xlabel("Keyword")

        ax.set_ylabel("Frequency")

        plt.xticks(rotation=70)

        plt.tight_layout()

        st.pyplot(fig)

        # ------------------------------------------
        # Top Topics
        # ------------------------------------------

        st.subheader("🏆 Top 10 Research Topics")

        st.write(

            ", ".join(

                trends["Keyword"].head(10)

            )

        )

        # ------------------------------------------
        # Rare Keywords
        # ------------------------------------------

        st.subheader("🌟 Rare Research Keywords")

        rare = trends[

            trends["Frequency"] == 1

        ]

        if not rare.empty:

            st.write(

                ", ".join(

                    rare["Keyword"].tolist()

                )

            )

        else:

            st.success(

                "No rare keywords found."

            )

        # ------------------------------------------
        # AI Summary
        # ------------------------------------------

        st.subheader("🤖 AI Research Summary")

        summary = f"""

Total Keywords Analysed : {total_keywords}

Unique Keywords : {unique_keywords}

Most Frequent Keyword : {top_keyword}

Frequency : {top_frequency}

Top Topics :

{', '.join(trends['Keyword'].head(10))}

"""

        st.info(summary)

        # ------------------------------------------
        # Download CSV
        # ------------------------------------------

        st.download_button(

            "⬇ Download Trend CSV",

            trends.to_csv(index=False),

            file_name="research_trends.csv",

            mime="text/csv"

        )

        # ------------------------------------------
        # Download Report
        # ------------------------------------------

        st.download_button(

            "⬇ Download Trend Report",

            summary,

            file_name="trend_report.txt",

            mime="text/plain"

        )

# =====================================================
# PAPER CLUSTERING
# =====================================================
with tab_cluster:

    st.header("🧩 Research Paper Clustering")

    documents = st.session_state.get("documents", [])

    if not st.session_state.get("database_ready", False):

        st.info("Please upload research papers first.")

    elif len(documents) < 2:

        st.warning("Upload at least two research papers for clustering.")

    else:

        doc_count = len(documents)

        # Maximum allowed clusters
        max_clusters = min(6, doc_count)

        # ----------------------------------------
        # Cluster Selection
        # ----------------------------------------

        if max_clusters <= 2:

            cluster_number = 2

            st.info(
                "Only two papers are available. Using **2 clusters** automatically."
            )

        else:

            cluster_number = st.slider(
                "Number of Clusters",
                min_value=2,
                max_value=max_clusters,
                value=min(3, max_clusters),
                step=1,
                key="cluster_slider"
            )

        # ----------------------------------------
        # Run Clustering
        # ----------------------------------------

        if st.button(
            "🚀 Generate Clusters",
            use_container_width=True,
            key="generate_clusters_btn"
        ):

            try:

                with st.spinner("Performing K-Means Clustering..."):

                    cluster_df = clustering.cluster(
                        documents,
                        embedder,
                        cluster_number
                    )

                # ----------------------------------------
                # Validate Output
                # ----------------------------------------

                if cluster_df is None:

                    st.error("Clustering returned no data.")
                    st.stop()

                if cluster_df.empty:

                    st.error("No clustering results were generated.")
                    st.stop()

                required_columns = {
                    "Paper",
                    "Cluster",
                    "X",
                    "Y"
                }

                missing = required_columns - set(cluster_df.columns)

                if missing:

                    st.error(
                        f"Missing required columns: {', '.join(missing)}"
                    )
                    st.stop()

                st.success(
                    "Paper Clustering Completed Successfully!"
                )

                # ----------------------------------------
                # Metrics
                # ----------------------------------------

                st.subheader("📊 Clustering Overview")

                actual_clusters = cluster_df["Cluster"].nunique()

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "📄 Papers",
                    len(cluster_df)
                )

                col2.metric(
                    "🧩 Clusters",
                    actual_clusters
                )

                col3.metric(
                    "📈 Avg Papers / Cluster",
                    round(
                        len(cluster_df) / max(actual_clusters, 1),
                        2
                    )
                )

                st.divider()

                # ----------------------------------------
                # Cluster Assignment
                # ----------------------------------------

                st.subheader("📋 Cluster Assignment")

                st.dataframe(
                    cluster_df,
                    use_container_width=True,
                    hide_index=True
                )

                # ----------------------------------------
                # Scatter Plot
                # ----------------------------------------

                st.subheader("📊 Cluster Visualization")

                fig, ax = plt.subplots(figsize=(8, 6))

                scatter = ax.scatter(
                    cluster_df["X"],
                    cluster_df["Y"],
                    c=cluster_df["Cluster"],
                    s=180
                )

                for _, row in cluster_df.iterrows():

                    ax.text(
                        row["X"],
                        row["Y"],
                        str(row["Paper"]),
                        fontsize=8
                    )

                ax.set_xlabel("PCA Component 1")
                ax.set_ylabel("PCA Component 2")
                ax.set_title("Research Paper Clusters")

                plt.colorbar(
                    scatter,
                    ax=ax,
                    label="Cluster"
                )

                st.pyplot(fig)

                plt.close(fig)

                # ----------------------------------------
                # Cluster Summary
                # ----------------------------------------

                st.subheader("📚 Cluster Summary")

                summary = (
                    cluster_df
                    .groupby("Cluster")["Paper"]
                    .apply(list)
                )

                for cluster_id, papers in summary.items():

                    with st.expander(
                        f"Cluster {cluster_id}"
                    ):

                        for paper in papers:

                            st.write(f"• {paper}")

                # ----------------------------------------
                # Cluster Distribution
                # ----------------------------------------

                st.subheader("📈 Cluster Distribution")

                distribution = (
                    cluster_df["Cluster"]
                    .value_counts()
                    .sort_index()
                )

                fig2, ax2 = plt.subplots(figsize=(7, 4))

                ax2.bar(
                    distribution.index.astype(str),
                    distribution.values
                )

                ax2.set_xlabel("Cluster")
                ax2.set_ylabel("Number of Papers")

                st.pyplot(fig2)

                plt.close(fig2)

                # ----------------------------------------
                # Download Report
                # ----------------------------------------

                csv = cluster_df.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇ Download Cluster Report",
                    data=csv,
                    file_name="paper_clusters.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except ValueError as e:

                st.error(f"Clustering failed: {e}")

            except Exception as e:

                st.exception(e)
# =====================================================
# AI RESEARCH ADVISOR
# =====================================================

with tab_advisor:

    st.header("🎓 AI Research Advisor")

    if not st.session_state.database_ready:

        st.info(
            "Please upload research papers first."
        )

    else:

        if st.button(
            "Generate Research Advice",
            use_container_width=True
        ):

            context = ""

            for doc in st.session_state.documents:

                context += f"""

Paper:

{doc['name']}

Content:

{doc['text']}

-----------------------------------

"""

            with st.spinner(
                "Analyzing research collection..."
            ):

                try:

                    advice = llm.research_advisor(
                        context
                    )

                except Exception as e:

                    advice = str(e)

            st.success(
                "Research Advice Generated!"
            )

            st.markdown(advice)

            st.download_button(

                "⬇ Download Research Advice",

                advice,

                file_name="research_advice.md",

                mime="text/markdown"

            )

# =====================================================
# SEARCH PAPERS TAB
# =====================================================

with tab_search:

    st.header("🔍 Search Research Papers")

    if not st.session_state.database_ready:

        st.info("Please upload research papers first.")

    else:

        col1, col2 = st.columns([3, 1])

        with col1:

            query = st.text_input(

                "Search across all uploaded papers",

                placeholder="Example: transformer, deep learning, CNN..."

            )

        with col2:

            top_k = st.slider(

                "Top Results",

                min_value=1,

                max_value=10,

                value=5

            )

        st.divider()

        if st.button(

            "🔎 Search",

            use_container_width=True

        ):

            if not query.strip():

                st.warning(

                    "Please enter a search query."

                )

            else:

                with st.spinner(

                    "Searching papers..."

                ):

                    query_embedding = embedder.embed_query(

                        query

                    )

                    results = st.session_state.database.search(

                        query_embedding,

                        top_k=top_k

                    )

                if results:

                    st.success(

                        f"{len(results)} matching results found."

                    )

                    # -----------------------------------
                    # Search Metrics
                    # -----------------------------------

                    col1, col2 = st.columns(2)

                    col1.metric(

                        "Results",

                        len(results)

                    )

                    col2.metric(

                        "Best Similarity",

                        f"{results[0]['similarity']:.2%}"

                    )

                    st.divider()

                    # -----------------------------------
                    # Best Match
                    # -----------------------------------

                    best = results[0]

                    st.subheader("🎯 Best Match")

                    st.success(

                        f"""
Paper: **{best['metadata']['source']}**

Page: **{best['metadata']['page']}**

Similarity: **{best['similarity']:.2%}**
"""

                    )

                    # -----------------------------------
                    # All Results
                    # -----------------------------------

                    st.subheader("📚 Search Results")

                    report = ""

                    for i, item in enumerate(results, start=1):

                        meta = item["metadata"]

                        similarity = item["similarity"]

                        with st.expander(

                            f"📄 Result {i} | {meta['source']} | {similarity:.2%}"

                        ):

                            st.write(

                                f"**Page:** {meta['page']}"

                            )

                            st.write(

                                f"**Chunk:** {meta['chunk_id']}"

                            )

                            st.write(

                                f"**Similarity:** {similarity:.2%}"

                            )

                            st.markdown("---")

                            st.write(

                                item["document"]

                            )

                        report += f"""

Result {i}

Paper:
{meta['source']}

Page:
{meta['page']}

Chunk:
{meta['chunk_id']}

Similarity:
{similarity:.2%}

Content:
{item['document']}

------------------------------------------------

"""

                    # -----------------------------------
                    # Download
                    # -----------------------------------

                    st.download_button(

                        "⬇ Download Search Results",

                        report,

                        file_name="search_results.txt",

                        mime="text/plain"

                    )

                else:

                    st.warning(

                        "No matching research papers found."

                    )
# =====================================================
# RESEARCH DASHBOARD
# =====================================================

with tab_dashboard:

    st.header("🏠 AI Research Assistant Dashboard")

    if not st.session_state.database_ready:

        st.info("Please upload research papers to begin.")

    else:

        # ----------------------------------------
        # Overall Statistics
        # ----------------------------------------

        total_papers = len(st.session_state.documents)

        total_pages = sum(

            len(doc["pages"])

            if isinstance(doc["pages"], list)

            else doc["pages"]

            for doc in st.session_state.documents

        )

        total_words = sum(

            len(doc["text"].split())

            for doc in st.session_state.documents

        )

        total_chunks = len(st.session_state.chunks)

        reading_time = round(total_words / 200, 1)

        col1, col2, col3, col4, col5 = st.columns(5)

        col1.metric("📄 Papers", total_papers)

        col2.metric("📚 Pages", total_pages)

        col3.metric("📝 Words", f"{total_words:,}")

        col4.metric("🧩 Chunks", total_chunks)

        col5.metric("⏱ Reading", f"{reading_time} min")

        st.divider()

        # ----------------------------------------
        # Paper Overview
        # ----------------------------------------

        st.subheader("📋 Paper Collection")

        overview = pd.DataFrame({

            "Paper": [

                doc["name"]

                for doc in st.session_state.documents

            ],

            "Pages": [

                len(doc["pages"])

                if isinstance(doc["pages"], list)

                else doc["pages"]

                for doc in st.session_state.documents

            ],

            "Words": [

                len(doc["text"].split())

                for doc in st.session_state.documents

            ]

        })

        st.dataframe(

            overview,

            use_container_width=True,

            hide_index=True

        )

        # ----------------------------------------
        # Paper Size Chart
        # ----------------------------------------

        st.subheader("📊 Paper Size Comparison")

        fig, ax = plt.subplots(figsize=(10,5))

        ax.bar(

            overview["Paper"],

            overview["Words"]

        )

        ax.set_ylabel("Words")

        plt.xticks(rotation=45)

        plt.tight_layout()

        st.pyplot(fig)

        # ----------------------------------------
        # Reading Progress
        # ----------------------------------------

        st.subheader("📖 Reading Progress")

        goal = 50000

        progress = min(

            total_words / goal,

            1.0

        )

        st.progress(progress)

        st.caption(

            f"{total_words:,} / {goal:,} words"

        )

        # ----------------------------------------
        # Collection Insights
        # ----------------------------------------

        st.subheader("💡 Collection Insights")

        largest = max(

            st.session_state.documents,

            key=lambda x: len(x["text"].split())

        )

        smallest = min(

            st.session_state.documents,

            key=lambda x: len(x["text"].split())

        )

        col1, col2 = st.columns(2)

        with col1:

            st.success(

                f"""
### 📈 Largest Paper

**{largest['name']}**

Words:

**{len(largest['text'].split()):,}**
"""

            )

        with col2:

            st.info(

                f"""
### 📉 Smallest Paper

**{smallest['name']}**

Words:

**{len(smallest['text'].split()):,}**
"""

            )

        st.divider()

        # ----------------------------------------
        # Quick Navigation
        # ----------------------------------------

        st.subheader("🚀 Available Features")

        features = [

            "💬 Chat with Papers",

            "📄 AI Summary",

            "📝 Quiz Generator",

            "🔍 Semantic Search",

            "📊 Statistics",

            "🔑 Keyword Analysis",

            "☁️ Word Cloud",

            "📈 Trend Analysis",

            "📊 Similarity Analysis",

            "🤖 AI Paper Comparison",

            "🧩 Paper Clustering",

            "🎓 AI Research Advisor",

            "📂 Metadata Viewer",

            "ℹ Paper Information"

        ]

        for feature in features:

            st.write("✅", feature)

        st.divider()

        # ----------------------------------------
        # Download Dashboard
        # ----------------------------------------

        dashboard_report = f"""
AI Research Assistant Dashboard

Total Papers : {total_papers}

Total Pages : {total_pages}

Total Words : {total_words}

Total Chunks : {total_chunks}

Reading Time : {reading_time} minutes

Largest Paper : {largest['name']}

Smallest Paper : {smallest['name']}
"""

        st.download_button(

            "⬇ Download Dashboard Report",

            dashboard_report,

            file_name="dashboard_report.txt",

            mime="text/plain"

        )
