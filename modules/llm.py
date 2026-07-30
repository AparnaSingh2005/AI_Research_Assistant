"""
llm.py

LLM interface for AI Research Paper Assistant
Groq Version

Supports

- Chat
- Summary
- Quiz
- Research Gap
- Key Insights
- Recommendations
- Keywords
- Metadata
- Citations
- Compare Papers
- Research Advisor
"""

import os
import time

from dotenv import load_dotenv
from groq import Groq

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

api_key = (
    st.secrets.get("GROQ_API_KEY")
    or os.getenv("GROQ_API_KEY")
)


class LLM:

    # ======================================================
    # Constructor
    # ======================================================

    def __init__(self):

        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:

            raise ValueError(
                "GROQ_API_KEY not found in .env"
            )

        self.client = Groq(
            api_key=api_key
        )

        # Best quality model
        self.model = "llama-3.3-70b-versatile"

        # Faster model (optional)
        # self.model = "llama-3.1-8b-instant"

    # ======================================================
    # Change Model
    # ======================================================

    def set_model(self, model_name):

        self.model = model_name

    # ======================================================
    # Internal LLM Call
    # ======================================================

    def _generate(self, prompt):

        try:

            # Prevent huge prompts
            MAX_CHARS = 8000
            if len(prompt) > MAX_CHARS:
                prompt = prompt[:MAX_CHARS]

            retries = 3

            delay = 2

            for attempt in range(retries):

                try:

                    response = self.client.chat.completions.create(

                        model=self.model,

                        messages=[

                            {
                                "role": "system",

                                "content":
                                """
You are an expert AI Research Assistant.

Always answer clearly.

Use Markdown formatting.

If the information is unavailable,
say so politely.

Never hallucinate facts.
"""
                            },

                            {
                                "role": "user",

                                "content": prompt
                            }

                        ],

                        temperature=0.3,

                        max_tokens=2048

                    )

                    return response.choices[0].message.content

                except Exception as e:

                    error = str(e)

                    if "429" in error:

                        print(
                            "Rate limit reached."
                        )

                        time.sleep(delay)

                        delay *= 2

                        continue

                    if "503" in error:

                        print(
                            "Groq temporarily unavailable."
                        )

                        time.sleep(delay)

                        delay *= 2

                        continue

                    raise

            return (
                "Error: Maximum retry limit reached."
            )

        except Exception as e:

            return f"Error: {e}"

    # ======================================================
    # Chat
    # ======================================================

    def ask(
        self,
        question,
        context
    ):

        prompt = f"""
You are an expert AI Research Assistant.

Answer ONLY using the supplied research papers.

If the answer is unavailable, reply exactly:

"I couldn't find this information in the uploaded research papers."

Research Papers

{context}

Question

{question}

Answer
"""

        return self._generate(prompt)

    # ======================================================
    # Summary
    # ======================================================

    def summarize(
        self,
        context
    ):

        prompt = f"""
Summarize the following research paper.

Include

• Objective

• Methodology

• Dataset

• Results

• Conclusion

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Quiz Generator
    # ======================================================

    def generate_quiz(

        self,

        context,

        difficulty="Medium",

        num_questions=10

    ):

        prompt = f"""
Generate {num_questions} multiple-choice questions from the research paper.

Difficulty Level:
{difficulty}

Instructions

• Each question should have exactly 4 options.

• Mention the correct answer after every question.

• Keep questions conceptual.

• Format using Markdown.

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Research Gap
    # ======================================================

    def research_gap(

        self,

        context

    ):

        prompt = f"""
Analyze the following research paper.

Identify

• Research gaps

• Current limitations

• Open research problems

• Possible future work

• Suggestions for improving the work

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Key Insights
    # ======================================================

    def key_insights(

        self,

        context

    ):

        prompt = f"""
Extract the most important insights.

Include

• Main contribution

• Novel ideas

• Important findings

• Methodology

• Real-world applications

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Recommendations
    # ======================================================

    def recommendations(

        self,

        context

    ):

        prompt = f"""
Based on this research paper suggest

• Similar research papers

• Future work

• Better algorithms

• Better datasets

• Technologies to explore

• Practical implementation ideas

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Keywords
    # ======================================================

    def keywords(

        self,

        context

    ):

        prompt = f"""
Extract the 20 most important keywords.

Rules

• Only one or two-word keywords

• No duplicates

• Return only bullet points

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Metadata Extraction
    # ======================================================

    def metadata(

        self,

        context

    ):

        prompt = f"""
Extract the following information.

Return exactly in this format.

Title:

Authors:

Year:

Journal / Conference:

DOI:

Research Area:

Keywords:

Abstract (3-4 lines):

Research Paper

{context}
"""

        return self._generate(prompt)

    # ======================================================
    # Citation Generator
    # ======================================================

    def citations(

        self,

        context

    ):

        prompt = f"""
Generate citations for the research paper.

Provide

APA

IEEE

MLA

BibTeX

Research Paper

{context}
"""

        return self._generate(prompt)



