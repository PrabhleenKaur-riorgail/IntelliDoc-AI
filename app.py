import streamlit as st
import pandas as pd
from azure_services import AzureAIHybridService

st.set_page_config(
    page_title="IntelliDoc AI | Azure AI-901 Project",
    page_icon="🔷",
    layout="wide"
)

ai_service = AzureAIHybridService()

with st.sidebar:
    st.title("🔷 IntelliDoc AI")
    st.caption("Aligned with Microsoft Azure AI Fundamentals (AI-901)")
    st.divider()

    st.subheader("⚙️ Active AI Services")
    status_map = ai_service.get_mode_status()
    for service_name, mode in status_map.items():
        st.markdown(f"**{service_name}**")
        st.info(mode)

    st.divider()
    st.subheader("🛡️ Responsible AI Principles")
    st.markdown("""
    - **Fairness**: Unbiased rule/model scoring
    - **Reliability & Safety**: Deterministic fallbacks
    - **Privacy & Security**: Automated PII masking
    - **Inclusiveness**: Accessible UI & clear outputs
    - **Transparency**: Confidence scores displayed
    - **Accountability**: Human-in-the-loop audit logs
    """)

st.title("IntelliDoc AI: Intelligent Document & Responsible AI Analyzer")
st.markdown(
    "An end-to-end enterprise AI pipeline integrating **Document Intelligence**, "
    "**Natural Language Processing (NLP)**, **Content Safety & PII Redaction**, and **Grounded Generative AI Q&A**."
)

col_input1, col_input2 = st.columns([1, 2])

with col_input1:
    st.subheader("1. Load Document")
    input_source = st.radio(
        "Choose input method:",
        ["Sample Invoice (Document Intelligence)", "Sample Customer Review (NLP & PII)", "Upload .txt File / Paste Text"]
    )

    sample_invoice = """INVOICE
Invoice Number: INV-2026-8841
Invoice Date: 2026-11-20
Vendor Name: Apex Cloud Solutions Ltd
Client Name: Mr. Rohan Sharma
Client Email: rohan.sharma@enterprise.in
Client Phone: +91-9876543210
Billing Account: 4532-8910-2234-9901

Line Items:
Item Description | Quantity | Unit Price | Total
Azure AI Compute Cluster | 2 | $450.00 | $900.00
Document Intelligence API Tier | 1 | $150.00 | $150.00
Content Safety Guardrail Add-on | 1 | $100.00 | $100.00

Subtotal: $1,150.00
Tax (18% GST): $207.00
Total Amount Due: $1,357.00
Payment Status: Approved
"""

    sample_feedback = """Customer Support Ticket #4092
Customer Name: Ms. Ananya Verma
Contact Email: ananya.verma99@gmail.com
Phone: 9811223344
PAN ID: ABCDE1234F
Date: 2026-11-19

Feedback Message:
I am extremely frustrated because my billing portal failed twice and I was overcharged $150.00 on 2026-11-19. However, the support engineer Mr. Vikram Mehta from Apex Cloud Solutions Ltd was very helpful and resolved the issue quickly. Overall, the recovery was smooth, though the initial delay was unacceptable.
"""

    uploaded_text = ""
    if input_source == "Sample Invoice (Document Intelligence)":
        uploaded_text = sample_invoice
    elif input_source == "Sample Customer Review (NLP & PII)":
        uploaded_text = sample_feedback
    else:
        uploaded_file = st.file_uploader("Upload a plain text (.txt) document", type=["txt"])
        if uploaded_file is not None:
            uploaded_text = uploaded_file.read().decode("utf-8")

with col_input2:
    st.subheader("2. Document Workspace")
    document_text = st.text_area(
        "Review or edit document text before running AI pipeline:",
        value=uploaded_text,
        height=260
    )

if not document_text.strip():
    st.warning("Please select a sample document or enter text above to run the AI pipeline.")
    st.stop()

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "📄 1. Document Intelligence (OCR)",
    "🧠 2. NLP & Text Analytics",
    "🛡️ 3. Responsible AI & Content Safety",
    "💬 4. Generative AI Document Q&A"
])

with tab1:
    st.subheader("Structured Field & Table Extraction")
    doc_res = ai_service.extract_document_fields(document_text)
    st.caption(f"Engine: {doc_res['source']}")

    m1, m2, m3 = st.columns(3)
    m1.metric("Lines Scanned", doc_res.get("total_lines", 0))
    m2.metric("Total Words", doc_res.get("total_words", 0))
    m3.metric("Key-Value Fields Extracted", len(doc_res.get("key_value_pairs", {})))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Extracted Key-Value Pairs")
        if doc_res.get("key_value_pairs"):
            kv_df = pd.DataFrame(list(doc_res["key_value_pairs"].items()), columns=["Field Name", "Extracted Value"])
            st.dataframe(kv_df, use_container_width=True, hide_index=True)
        else:
            st.info("No explicit Key: Value pairs found.")

    with c2:
        st.markdown("#### Extracted Line Items / Tables")
        if doc_res.get("line_items"):
            st.dataframe(pd.DataFrame(doc_res["line_items"]), use_container_width=True, hide_index=True)
        else:
            st.info("No structured table rows detected in this document.")

with tab2:
    st.subheader("Azure AI Language Capabilities")
    nlp_res = ai_service.analyze_nlp(document_text)
    st.caption(f"Engine: {nlp_res['source']}")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"**Detected Language:** `{nlp_res['language']}`")
        st.markdown(f"**Overall Document Sentiment:** `{nlp_res['sentiment']}`")
        st.markdown("#### Sentiment Confidence Scores")
        scores_df = pd.DataFrame({
            "Sentiment": list(nlp_res["confidence_scores"].keys()),
            "Confidence": list(nlp_res["confidence_scores"].values())
        })
        st.bar_chart(scores_df.set_index("Sentiment"))

    with col_b:
        st.markdown("#### Key Phrase Extraction")
        if nlp_res["key_phrases"]:
            st.write(", ".join([f"`{kp}`" for kp in nlp_res["key_phrases"]]))
        else:
            st.write("No key phrases extracted.")

        st.markdown("#### Named Entity Recognition (NER)")
        if nlp_res["entities"]:
            st.dataframe(pd.DataFrame(nlp_res["entities"]), use_container_width=True, hide_index=True)
        else:
            st.info("No named entities detected.")

with tab3:
    st.subheader("Content Moderation & PII Privacy Guardrails")
    safety_res = ai_service.check_responsible_ai(document_text)
    st.caption(f"Engine: {safety_res['source']}")

    if safety_res["is_safe"]:
        st.success("✅ Content Safety Check Passed: Zero harmful content detected across Hate, Violence, Self-Harm, and Sexual categories.")
    else:
        st.error("⚠️ Content Safety Alert: Potential policy violation detected.")

    s_cols = st.columns(4)
    for idx, (cat, sev) in enumerate(safety_res["severity_scores"].items()):
        s_cols[idx].metric(f"{cat} Severity (0–6)", sev)

    st.markdown("---")
    st.markdown(f"#### Personally Identifiable Information (PII) Redaction ({safety_res['pii_count']} items masked)")
    if safety_res["pii_findings"]:
        st.dataframe(pd.DataFrame(safety_res["pii_findings"]), use_container_width=True, hide_index=True)

    st.markdown("#### Sanitized Output (Safe for Downstream LLM / Storage)")
    st.code(safety_res["redacted_text"], language="text")

with tab4:
    st.subheader("Grounded Document Assistant (RAG Pattern)")
    st.markdown("Ask questions grounded strictly in the uploaded document to prevent hallucinations.")
    user_q = st.text_input("Enter your question (e.g., 'What is the total amount due?' or 'Who resolved the issue?'):")
    if user_q:
        answer = ai_service.answer_document_question(document_text, user_q)
        st.markdown(answer)
