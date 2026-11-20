import os
import re
from typing import Dict, List, Any
from dotenv import load_dotenv

# Official Azure AI SDK Imports (demonstrates genuine Azure SDK integration)
try:
    from azure.core.credentials import AzureKeyCredential
    from azure.ai.documentintelligence import DocumentIntelligenceClient
    from azure.ai.textanalytics import TextAnalyticsClient
    from azure.ai.contentsafety import ContentSafetyClient
    from azure.ai.contentsafety.models import AnalyzeTextOptions
    from openai import AzureOpenAI
    AZURE_SDKS_INSTALLED = True
except ImportError:
    AZURE_SDKS_INSTALLED = False

load_dotenv()


class AzureAIHybridService:
    """
    Unified service layer mapping directly to AI-901 domains:
    1. Document Intelligence (OCR & Key-Value Extraction)
    2. Azure AI Language (Sentiment, Key Phrases, NER, Language Detection)
    3. Azure AI Content Safety & Responsible AI (PII Redaction & Moderation)
    4. Azure OpenAI / Microsoft Foundry (Grounded Document Q&A / RAG)

    Automatically uses live Azure endpoints if configured in .env, otherwise
    falls back cleanly to a deterministic local engine for offline/demo reliability.
    """

    def __init__(self):
        self.doc_endpoint = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT", "").strip()
        self.doc_key = os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY", "").strip()

        self.lang_endpoint = os.getenv("AZURE_LANGUAGE_ENDPOINT", "").strip()
        self.lang_key = os.getenv("AZURE_LANGUAGE_KEY", "").strip()

        self.safety_endpoint = os.getenv("AZURE_CONTENT_SAFETY_ENDPOINT", "").strip()
        self.safety_key = os.getenv("AZURE_CONTENT_SAFETY_KEY", "").strip()

        self.oai_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").strip()
        self.oai_key = os.getenv("AZURE_OPENAI_KEY", "").strip()
        self.oai_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o").strip()

    def get_mode_status(self) -> Dict[str, str]:
        return {
            "Document Intelligence": "Live Azure Cloud" if (self.doc_endpoint and self.doc_key) else "Hybrid Local Engine",
            "Azure AI Language (NLP)": "Live Azure Cloud" if (self.lang_endpoint and self.lang_key) else "Hybrid Local Engine",
            "Azure AI Content Safety": "Live Azure Cloud" if (self.safety_endpoint and self.safety_key) else "Hybrid Local Engine",
            "Azure OpenAI (GenAI)": "Live Azure Cloud" if (self.oai_endpoint and self.oai_key) else "Hybrid Local Engine",
        }

    # ---------------------------------------------------------
    # 1. AZURE AI DOCUMENT INTELLIGENCE (OCR & Field Extraction)
    # ---------------------------------------------------------
    def extract_document_fields(self, text: str) -> Dict[str, Any]:
        if AZURE_SDKS_INSTALLED and self.doc_endpoint and self.doc_key:
            client = DocumentIntelligenceClient(
                endpoint=self.doc_endpoint,
                credential=AzureKeyCredential(self.doc_key)
            )
            poller = client.begin_analyze_document("prebuilt-invoice", body=text.encode("utf-8"))
            result = poller.result()
            return {"source": "Azure Document Intelligence API", "raw_content": result.content}

        # Hybrid Local Extraction Engine (Key-Value & Table Parser)
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        kv_pairs = {}
        line_items = []

        for line in lines:
            if ":" in line and not line.lower().startswith("http"):
                key, val = line.split(":", 1)
                if len(key.strip()) <= 35 and val.strip():
                    kv_pairs[key.strip()] = val.strip()

            if "|" in line:
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 3 and not parts[0].lower().startswith("item"):
                    line_items.append({
                        "Item Description": parts[0],
                        "Quantity": parts[1] if len(parts) > 1 else "1",
                        "Unit Price": parts[2] if len(parts) > 2 else "-",
                        "Total": parts[3] if len(parts) > 3 else parts[-1]
                    })

        dates = re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|[A-Z][a-z]+ \d{1,2}, \d{4})\b", text)
        amounts = re.findall(r"(?:\$|INR|Rs\.?)\s?\d+(?:,\d{3})*(?:\.\d{2})?", text)

        if "Detected Dates" not in kv_pairs and dates:
            kv_pairs["Detected Dates"] = ", ".join(list(dict.fromkeys(dates)))
        if "Monetary Amounts" not in kv_pairs and amounts:
            kv_pairs["Monetary Amounts"] = ", ".join(list(dict.fromkeys(amounts)))

        return {
            "source": "Azure AI Document Intelligence (Hybrid Local Engine)",
            "total_lines": len(lines),
            "total_words": len(text.split()),
            "key_value_pairs": kv_pairs,
            "line_items": line_items
        }

    # ---------------------------------------------------------
    # 2. AZURE AI LANGUAGE (Sentiment, NER, Key Phrases, Lang)
    # ---------------------------------------------------------
    def analyze_nlp(self, text: str) -> Dict[str, Any]:
        if AZURE_SDKS_INSTALLED and self.lang_endpoint and self.lang_key:
            client = TextAnalyticsClient(
                endpoint=self.lang_endpoint,
                credential=AzureKeyCredential(self.lang_key)
            )
            lang_resp = client.detect_language([text])[0]
            sent_resp = client.analyze_sentiment([text])[0]
            kp_resp = client.extract_key_phrases([text])[0]
            ner_resp = client.recognize_entities([text])[0]
            return {
                "source": "Azure AI Language API",
                "language": lang_resp.primary_language.name,
                "sentiment": sent_resp.sentiment,
                "confidence_scores": {
                    "positive": sent_resp.confidence_scores.positive,
                    "neutral": sent_resp.confidence_scores.neutral,
                    "negative": sent_resp.confidence_scores.negative,
                },
                "key_phrases": list(kp_resp.key_phrases),
                "entities": [{"text": e.text, "category": e.category, "confidence": e.confidence_score} for e in ner_resp.entities]
            }

        positive_words = {"excellent", "great", "fast", "helpful", "smooth", "satisfied", "love", "resolved", "amazing", "efficient", "good", "approved"}
        negative_words = {"delay", "delayed", "broken", "terrible", "slow", "issue", "error", "frustrated", "bad", "poor", "overcharged", "unacceptable", "failed"}

        words_clean = re.findall(r"\b[a-zA-Z]+\b", text.lower())
        pos_hits = sum(1 for w in words_clean if w in positive_words)
        neg_hits = sum(1 for w in words_clean if w in negative_words)

        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            sentiment = "Neutral"
            scores = {"Positive": 0.15, "Neutral": 0.75, "Negative": 0.10}
        elif pos_hits > neg_hits:
            sentiment = "Positive"
            pos_ratio = round(min(0.95, 0.55 + (pos_hits - neg_hits) * 0.1), 2)
            scores = {"Positive": pos_ratio, "Neutral": round((1 - pos_ratio) * 0.7, 2), "Negative": round((1 - pos_ratio) * 0.3, 2)}
        elif neg_hits > pos_hits:
            sentiment = "Negative"
            neg_ratio = round(min(0.95, 0.55 + (neg_hits - pos_hits) * 0.1), 2)
            scores = {"Positive": round((1 - neg_ratio) * 0.2, 2), "Neutral": round((1 - neg_ratio) * 0.8, 2), "Negative": neg_ratio}
        else:
            sentiment = "Mixed"
            scores = {"Positive": 0.45, "Neutral": 0.10, "Negative": 0.45}

        entities = []
        for org in set(re.findall(r"\b[A-Z][a-zA-Z]+ (?:Corp|Inc|Ltd|Technologies|Solutions|Systems|University|Bank)\b", text)):
            entities.append({"Entity": org, "Category": "Organization", "Confidence": 0.94})
        for person in set(re.findall(r"\b(?:Mr\.|Ms\.|Dr\.) [A-Z][a-z]+ [A-Z][a-z]+\b", text)):
            entities.append({"Entity": person, "Category": "Person", "Confidence": 0.92})
        for money in set(re.findall(r"(?:\$|INR|Rs\.?)\s?\d+(?:,\d{3})*(?:\.\d{2})?", text)):
            entities.append({"Entity": money, "Category": "Quantity / Currency", "Confidence": 0.98})
        for date in set(re.findall(r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4})\b", text)):
            entities.append({"Entity": date, "Category": "DateTime", "Confidence": 0.96})

        stopwords = {"the", "and", "is", "in", "to", "of", "for", "with", "on", "at", "from", "by", "this", "that", "was", "are", "be", "have", "has", "had"}
        freq = {}
        for w in words_clean:
            if len(w) > 4 and w not in stopwords:
                freq[w] = freq.get(w, 0) + 1
        key_phrases = [k.capitalize() for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:8]]

        return {
            "source": "Azure AI Language (Hybrid Local Engine)",
            "language": "English (en)",
            "sentiment": sentiment,
            "confidence_scores": scores,
            "key_phrases": key_phrases,
            "entities": entities
        }

    # ---------------------------------------------------------
    # 3. RESPONSIBLE AI & AZURE AI CONTENT SAFETY (PII & Guardrails)
    # ---------------------------------------------------------
    def check_responsible_ai(self, text: str) -> Dict[str, Any]:
        if AZURE_SDKS_INSTALLED and self.safety_endpoint and self.safety_key:
            client = ContentSafetyClient(
                endpoint=self.safety_endpoint,
                credential=AzureKeyCredential(self.safety_key)
            )
            response = client.analyze_text(AnalyzeTextOptions(text=text[:1000]))
            severities = {item.category: item.severity for item in response.categories_analysis}
        else:
            lowered = text.lower()
            severities = {
                "Hate": 4 if any(w in lowered for w in ["hate", "discriminate", "slur"]) else 0,
                "SelfHarm": 4 if any(w in lowered for w in ["self-harm", "suicide"]) else 0,
                "Sexual": 4 if any(w in lowered for w in ["explicit", "nsfw"]) else 0,
                "Violence": 4 if any(w in lowered for w in ["attack", "kill", "violent", "threat"]) else 0,
            }

        pii_findings = []
        redacted_text = text

        patterns = {
            "Email Address": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "Phone Number": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
            "Credit Card / Account ID": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
            "Government ID / SSN / PAN": r"\b(?:\d{3}-\d{2}-\d{4}|[A-Z]{5}\d{4}[A-Z])\b",
        }

        for pii_type, pattern in patterns.items():
            matches = re.findall(pattern, text)
            for m in matches:
                pii_findings.append({"PII Type": pii_type, "Detected Value": m, "Action": "Masked"})
                redacted_text = redacted_text.replace(m, f"[REDACTED-{pii_type.upper()}]")

        overall_safe = all(v == 0 for v in severities.values())

        return {
            "source": "Azure AI Content Safety & PII Guardrail",
            "is_safe": overall_safe,
            "severity_scores": severities,
            "pii_count": len(pii_findings),
            "pii_findings": pii_findings,
            "redacted_text": redacted_text
        }

    # ---------------------------------------------------------
    # 4. AZURE OPENAI / MICROSOFT FOUNDRY (Grounded Document Q&A)
    # ---------------------------------------------------------
    def answer_document_question(self, document_text: str, question: str) -> str:
        if AZURE_SDKS_INSTALLED and self.oai_endpoint and self.oai_key:
            client = AzureOpenAI(
                azure_endpoint=self.oai_endpoint,
                api_key=self.oai_key,
                api_version="2024-06-01"
            )
            response = client.chat.completions.create(
                model=self.oai_deployment,
                messages=[
                    {"role": "system", "content": "You are an enterprise AI assistant grounded strictly in the provided document. Cite facts accurately."},
                    {"role": "user", "content": f"Document:\n{document_text}\n\nQuestion: {question}"}
                ],
                temperature=0.2
            )
            return response.choices[0].message.content

        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", document_text) if s.strip()]
        q_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", question.lower())) - {"what", "when", "where", "who", "how", "the", "for", "are", "was", "this", "that"}

        if not q_tokens:
            return "Please ask a specific question about dates, amounts, names, items, or issues mentioned in the document."

        scored = []
        for s in sentences:
            s_tokens = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", s.lower()))
            overlap = len(q_tokens.intersection(s_tokens))
            if overlap > 0:
                scored.append((overlap, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        if not scored:
            return "Grounded AI Response: The uploaded document does not contain specific information answering that query (Hallucination Prevention Guardrail triggered)."

        top_evidence = [s for _, s in scored[:3]]
        return (
            "**Grounded Answer (Retrieved from Document Context):**\n\n"
            + "\n".join([f"- {ev}" for ev in top_evidence])
        )
