import os
from groq import Groq
from vector_store import query_store
from config import LLM_MODEL
from sentiment import detect_sentiment
from translation import detect_language, translate_to_english, translate_from_english

DISTANCE_THRESHOLD = 1.0

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
GROQ_MODEL = "openai/gpt-oss-20b"


def build_prompt(question, retrieved_chunks, sentiment):
    context = "\n\n".join(
        f"[Source: {chunk_meta['source']}]\n{chunk_text}"
        for chunk_text, chunk_meta in retrieved_chunks
    )

    tone_instruction = {
        "negative": "The user seems frustrated or unhappy. Respond with extra patience, empathy, and clarity.",
        "positive": "The user seems happy or pleased. Keep your tone warm and matching their energy.",
        "neutral": "Respond in a clear, professional, helpful tone."
    }[sentiment]

    prompt = f"""You are a helpful assistant. {tone_instruction}
Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I don't have information about that in my knowledge base."

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def get_rag_response(question, n_results=3):
    user_lang = detect_language(question)
    question_en = translate_to_english(question, user_lang)

    sentiment_result = detect_sentiment(question_en)
    sentiment = sentiment_result["sentiment"]

    results = query_store(question_en, n_results=n_results)
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered = [
        (doc, meta) for doc, meta, dist in zip(documents, metadatas, distances)
        if dist <= DISTANCE_THRESHOLD
    ]

    if not filtered:
        fallback = "I don't have information about that in my knowledge base."
        return {
            "answer": translate_from_english(fallback, user_lang),
            "sources": [],
            "sentiment": sentiment,
            "language": user_lang
        }

    prompt = build_prompt(question_en, filtered, sentiment)

    response = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": prompt}]
    )
    answer_en = response.choices[0].message.content

    answer = translate_from_english(answer_en, user_lang)

    sources = list(set(meta["source"] for _, meta in filtered))
    return {
        "answer": answer,
        "sources": sources,
        "sentiment": sentiment,
        "language": user_lang
    }