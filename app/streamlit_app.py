from __future__ import annotations

import os
from typing import Any

import pandas as pd
import requests
import streamlit as st


DEFAULT_API_URL = os.getenv("TICKET_API_URL", "http://127.0.0.1:8000")
EXAMPLE_TEXT = "My bank charged me twice for the same transaction and has not refunded the duplicate charge."


def api_post(base_url: str, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}", json=payload, timeout=120)
    response.raise_for_status()
    return response.json()


def api_get(base_url: str, endpoint: str) -> dict[str, Any]:
    response = requests.get(f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}", timeout=30)
    response.raise_for_status()
    return response.json()


def filters_payload(product: str, company: str, issue: str, state: str) -> dict[str, str]:
    filters = {
        "product": product.strip(),
        "company": company.strip(),
        "issue": issue.strip(),
        "state": state.strip().upper(),
    }
    return {key: value for key, value in filters.items() if value}


def show_search_results(results: list[dict[str, Any]]) -> None:
    if not results:
        st.info("No matching complaints found for the current query and filters.")
        return

    table = pd.DataFrame(
        [
            {
                "rank": item.get("rank"),
                "similarity": item.get("similarity"),
                "complaint_id": item.get("complaint_id"),
                "product": item.get("product"),
                "issue": item.get("issue"),
                "company": item.get("company"),
                "outcome": item.get("company_response"),
            }
            for item in results
        ]
    )
    st.dataframe(table, use_container_width=True, hide_index=True)

    for item in results:
        title = f"#{item.get('rank')} · {item.get('product')} · {item.get('complaint_id')}"
        with st.expander(title):
            st.write(item.get("snippet", ""))
            st.caption(
                f"Issue: {item.get('issue', '')} | Company: {item.get('company', '')} | "
                f"Outcome: {item.get('company_response') or 'not available'} | "
                f"Timely: {item.get('timely_response') or 'not available'}"
            )


def main() -> None:
    st.set_page_config(page_title="Ticket Intelligence", layout="wide")
    st.title("Customer Support Ticket Intelligence")
    st.caption("Classify, search, summarize, and inspect similar CFPB complaints.")

    with st.sidebar:
        st.header("Service")
        api_url = st.text_input("API URL", value=DEFAULT_API_URL)
        try:
            health = api_get(api_url, "/health")
            metadata = api_get(api_url, "/metadata")
            st.success(f"API online: {health.get('version')}")
            st.metric("Dataset rows", metadata.get("dataset_rows") or 0)
            st.metric("FAISS rows", metadata.get("faiss_rows") or 0)
            products = [""] + metadata.get("products", [])
        except Exception as exc:
            st.error(f"API not reachable: {exc}")
            products = [""]

        st.header("Filters")
        product = st.selectbox("Product", products)
        company = st.text_input("Company contains")
        issue = st.text_input("Issue contains")
        state = st.text_input("State", max_chars=2)
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5)
        fetch_k = st.slider("Fetch before filtering", min_value=25, max_value=300, value=100, step=25)

    query = st.text_area("Complaint or search query", value=EXAMPLE_TEXT, height=150)
    filters = filters_payload(product, company, issue, state)

    tab_analyze, tab_rag, tab_search, tab_classify, tab_summarize = st.tabs(
        ["Analyze", "Retrieval Answer", "Search", "Classify", "Summarize"]
    )

    with tab_analyze:
        st.subheader("Analyze all")
        include_summary = st.checkbox("Include summary", value=False)
        if st.button("Run analysis", type="primary"):
            payload = {
                "text": query,
                "top_k": top_k,
                "filters": filters,
                "include_summary": include_summary,
            }
            try:
                data = api_post(api_url, "/analyze", payload)
                classification = data["classification"]
                st.metric("Predicted product", classification["predicted_product"])
                if classification.get("confidence") is not None:
                    st.metric("Classification confidence", classification["confidence"])

                rag = data["rag"]
                st.markdown("#### Retrieval answer")
                st.text(rag["answer"])
                show_search_results(rag.get("context", []))

                if data.get("summary"):
                    st.markdown("#### Summary")
                    st.write(data["summary"]["summary"])
            except Exception as exc:
                st.error(str(exc))

    with tab_rag:
        st.subheader("Retrieval-grounded answer")
        if st.button("Generate retrieval answer"):
            payload = {"query": query, "top_k": top_k, "fetch_k": fetch_k, "filters": filters}
            try:
                data = api_post(api_url, "/rag", payload)
                col1, col2, col3 = st.columns(3)
                col1.metric("Confidence", data["confidence"])
                col2.metric("Top score", data["top_score"])
                col3.metric("Retrieved", data["retrieved_count"])
                st.text(data["answer"])
                show_search_results(data.get("context", []))
            except Exception as exc:
                st.error(str(exc))

    with tab_search:
        st.subheader("Similar complaints")
        if st.button("Search complaints"):
            payload = {"query": query, "top_k": top_k, "fetch_k": fetch_k, "filters": filters}
            try:
                data = api_post(api_url, "/search", payload)
                st.caption(data.get("filters", {}))
                show_search_results(data.get("results", []))
            except Exception as exc:
                st.error(str(exc))

    with tab_classify:
        st.subheader("Complaint classification")
        if st.button("Classify complaint"):
            try:
                data = api_post(api_url, "/classify", {"text": query})
                st.metric("Predicted product", data["predicted_product"])
                if data.get("confidence") is not None:
                    st.metric("Confidence", data["confidence"])
                if data.get("class_scores"):
                    st.dataframe(pd.DataFrame(data["class_scores"]), use_container_width=True, hide_index=True)
            except Exception as exc:
                st.error(str(exc))

    with tab_summarize:
        st.subheader("Summarize complaint")
        complaint_id = st.text_input("Complaint ID optional")
        max_tokens = st.slider("Max summary tokens", min_value=30, max_value=220, value=110)
        min_tokens = st.slider("Min summary tokens", min_value=5, max_value=120, value=35)
        if st.button("Summarize"):
            payload: dict[str, Any] = {
                "complaint_id": complaint_id.strip() or None,
                "text": None if complaint_id.strip() else query,
                "max_summary_tokens": max_tokens,
                "min_summary_tokens": min_tokens,
            }
            try:
                data = api_post(api_url, "/summarize", payload)
                st.write(data["summary"])
                col1, col2, col3 = st.columns(3)
                col1.metric("Input words", data["input_word_count"])
                col2.metric("Summary words", data["summary_word_count"])
                col3.metric("Compression", data["compression_ratio"])
            except Exception as exc:
                st.error(str(exc))


if __name__ == "__main__":
    main()
