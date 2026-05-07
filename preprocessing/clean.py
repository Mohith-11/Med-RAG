# Cleaning preprocessing module
import re

def clean_text(text):
    text = re.sub(r'\n+', ' ', text)         # remove line breaks
    text = re.sub(r'\s+', ' ', text)         # normalize spaces
    text = text.strip()
    return text

def clean_pages(pages):
    return [clean_text(p) for p in pages]