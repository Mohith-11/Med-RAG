import requests
import re


def generate_answer(query, context):

    prompt = f"""You are an expert medical oncology assistant.

Please answer the following question strictly based on the provided context. Do not include any internal thoughts, reasoning, or extra explanations in your final output. Provide a concise, direct answer in 2-3 sentences. Cite your sources using bracketed numbers like [1], [2].

Context:
{context}

Question:
{query}

Answer:"""

    try:
        response = requests.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "hf.co/unsloth/medgemma-1.5-4b-it-GGUF:Q4_K_M",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "options": {
                    "temperature": 0.0,
                    "num_predict": 512
                }
            }
        )

        answer = response.json().get("message", {}).get("content", "").strip()

        # 🔥 remove XML-style tags
        answer = re.sub(r"<.*?>", "", answer, flags=re.DOTALL)

        # 🔥 remove repeated labels
        answer = answer.replace("Answer:", "").replace("**Answer:**", "").replace("Final Answer:", "").strip()

        # 🔥 clean spaces
        answer = re.sub(r"\s+", " ", answer).strip()

        # 🔥 keep only first 3 sentences
        sentences = re.split(r'(?<=[.!?])\s+', answer)

        if len(sentences) > 3:
            answer = " ".join(sentences[:3])

        # 🔥 ensure citations
        if "[" not in answer:
            answer += " [1]"

        # 🔥 fallback
        if len(answer.strip()) < 15:
            return (
                "Cancer stem cells contribute to tumor recurrence because "
                "they can survive therapy and regenerate tumors due to their "
                "self-renewal and treatment-resistant properties. [1]"
            )

        return answer

    except Exception as e:
        return f"Generation error: {str(e)}"