import subprocess
import sys
import numpy
import language_tool_python


tool = language_tool_python.LanguageToolPublicAPI('en-US')

text = "This are bad grammar."
matches = tool.check(text)

for match in matches:
    print(match)

subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])


import streamlit as st
from transformers import pipeline
import language_tool_python
import spacy

# --- Load models ---
translator_en_to_de = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-en-de",
    tokenizer="Helsinki-NLP/opus-mt-en-de"
)
translator_de_to_en = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-de-en",
    tokenizer="Helsinki-NLP/opus-mt-de-en"
)
lang_tool = language_tool_python.LanguageTool('de')
nlp = spacy.load("de_core_news_sm")

# --- Streamlit UI ---
st.set_page_config(page_title="AI Study Assistant - German Helper")
st.title("📘 AI Study Assistant for German Learners")

st.markdown("""
This app helps you:
- Correct German sentences
- Translate between English and German
- Learn vocabulary
""")

user_input = st.text_area("✍️ Enter a German or English sentence:")
action = st.selectbox("What do you want to do?", ["Correct Grammar (DE)", "Translate (EN ↔ DE)", "Extract Vocabulary (DE)"])

# --- Functions ---
def correct_grammar(text):
    matches = lang_tool.check(text)
    corrected = language_tool_python.utils.correct(text, matches)
    return corrected

def translate(text):
    # Detect language by checking for umlauts and ß (rough heuristic)
    direction = "de_to_en" if any(c in text for c in "äöüß") else "en_to_de"
    translator = translator_de_to_en if direction == "de_to_en" else translator_en_to_de
    result = translator(text)
    return result[0]['translation_text']

def extract_vocab(text):
    doc = nlp(text)
    vocab_items = [(token.text, token.pos_) for token in doc if token.pos_ in ["NOUN", "VERB", "ADJ"]]
    return vocab_items

# --- Handle Actions ---
if st.button("Run") and user_input:
    with st.spinner("Processing..."):
        if action == "Correct Grammar (DE)":
            result = correct_grammar(user_input)
            st.success("✅ Corrected Sentence:")
            st.write(result)
        elif action == "Translate (EN ↔ DE)":
            result = translate(user_input)
            st.success("🌐 Translated Sentence:")
            st.write(result)
        elif action == "Extract Vocabulary (DE)":
            vocab = extract_vocab(user_input)
            st.success("🧠 Vocabulary List:")
            for word, pos in vocab:
                st.write(f"- **{word}** ({pos})")

st.markdown("---")
st.markdown("Made with ❤️ for German learners. Uses Hugging Face, LanguageTool, and spaCy.")
