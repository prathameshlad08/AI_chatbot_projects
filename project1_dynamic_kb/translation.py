from langdetect import detect
from deep_translator import GoogleTranslator

def detect_language(text):
    try:
        return detect(text)
    except Exception:
        return "en"

def translate_to_english(text, source_lang):
    if source_lang == "en":
        return text
    try:
        return GoogleTranslator(source=source_lang, target="en").translate(text)
    except Exception:
        return text

def translate_from_english(text, target_lang):
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

if __name__ == "__main__":
    tests = [
        "¿Dónde está la Torre Eiffel?",
        "Qu'est-ce que la photosynthèse?",
        "What is the Great Wall of China?"
    ]
    for t in tests:
        lang = detect_language(t)
        en = translate_to_english(t, lang)
        print(f"Original ({lang}): {t}")
        print(f"English: {en}")
        print("---")