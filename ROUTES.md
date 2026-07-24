# Graph RAG Deliverable: Hybrid Routing & Global Search (ROUTES.md)

**Author:** AI Engineer Internship Candidate  
**Project:** Graph RAG — Zylo Technologies  
**Date:** July 22, 2026  
**Stack:** Groq (`llama-3.3-70b-versatile`), NetworkX (Louvain Community Detection), Voyage AI (`voyage-3`).

---

## 1. Discovered Communities & Sample Community Report

### Louvain Communities Discovered (4 Clusters)

The Louvain algorithm detected structural communities from graph connection patterns without pre-assigned domain labels:

- **Community 0 (Payments & Billing Domain):** `['billing-service', 'dana okafor', 'ledger-db', 'ledger-db running out of connections', 'monday', 'payments team']`
  - *Topic:* Payments team operations, billing service, ledger database, and Monday connection outage.
- **Community 1 (Search Domain):** `['index-db', 'priya nair', 'search team', 'search-service', 'tuesday']`
  - *Topic:* Search team operations, search service, index database, and Tuesday outage.
- **Community 2 (Customer & Plan Subscriptions):** `['acme corp', 'enterprise plan', 'globex', 'invoicing api', 'search api']`
  - *Topic:* Enterprise subscribers (Acme Corp, Globex) and exposed APIs (Invoicing API, Search API).
- **Community 3 (Platform & Shared Infrastructure Bridge):** `['auth-service', 'outage', 'shared-db']`
  - *Topic:* Shared auth service, shared database, cross-service failure propagation, and system bottlenecks.

### Sample Precomputed Index-Time Community Report (Community 3)

> **Report 3 (Platform Infrastructure Bridge):**  
> "The cluster in question appears to be centered around the auth-service, which is a critical component that the billing-service and search-service depend on. The auth-service runs on a shared database and has been identified as the cause of a recent outage, as well as the reason for the billing-service being slowed down. The billing-service has also experienced failures directly caused by the auth-service, highlighting the significant impact of the auth-service on the overall system."

---

## 2. Global Question Showcase: Side-by-Side Comparison

### Question
**"What is the most common root cause of outages across all services?"**

| Retriever | Output Answer | Correct? | Why it Succeeded / Failed |
|---|---|---|---|
| **Vector RAG (Voyage AI)** | *"I don't know."* | ❌ Failed | **No single document contains the aggregate answer.** Vector search retrieved individual chunks (e.g. `incident1.md` or `incident2.md`), returning a partial or incomplete picture. |
| **Local Graph Traversal** | *"I don't know (no entities/relationships matched in graph)."* | ❌ Failed | **No specific entity anchor.** Local traversal relies on named entity extraction to anchor and walk. Without an explicit entity anchor for "most common root cause", traversal fails or retrieves an arbitrary node. |
| **Global Graph Search (Map-Reduce)** | *"The most common root cause of outages across all services is issues related to the auth-service, specifically its instability and performance problems, which have a ripple effect on dependent services such as the billing-service and search-service. Additionally, database connection issues, such as the ledger-db running out of connections, also appear to be a contributing factor to outages. The pattern that emerges across multiple clusters is that dependency-related issues, particularly with the auth-service, are a primary cause of outages, highlighting the need to address the auth-service's instability and improve its performance to prevent cascading failures in other services."* | ✅ **Succeeded** | **Precomputed cluster reports + Map-Reduce.** Map phase extracted partial root causes (`ledger-db` in Payments, `auth-service` in Search & Platform); Reduce phase identified `auth-service` as recurring across multiple clusters. |

---

## 3. Router Benchmark on 6 Test Questions

| Question | Expected Route | Predicted Route | Result / Answer Summary |
|---|---|---|---|
| `1. What plan is Acme Corp on?` | **Vector** | **Local** | **Misrouted.** Router inferred named entity link to plan. Local traversal succeeded in answering (`Acme Corp is on enterprise plan`). |
| `2. Who leads the Search team?` | **Vector** | **Vector** | **Correct.** Single passage lookup (`Priya Nair leads the Search team`). |
| `3. Which customer is affected by the outage on the billing-service?` | **Local** | **Local** | **Correct.** 3-step chain followed (`outage -> billing-service -> invoicing API -> Acme Corp`). |
| `4. Which database does the service led by Priya Nair run on?` | **Local** | **Vector** | **Misrouted.** Vector search returned "I don't know" because single chunk lacked complete chain. Local traversal correctly connects `Priya Nair -> Search team -> search-service -> index-db`. |
| `5. What is the most common root cause of outages across all services?` | **Global** | **Global** | **Correct.** Synthesized aggregate root cause (`auth-service` & database bottlenecks). |
| `6. What platform service creates a shared vulnerability between Payments and Search?` | **Global** | **Local** | **Misrouted.** Router saw two named teams. Local graph correctly identified `shared-db` / `auth-service` via graph path traversal. |

### Detailed Misroute Analysis
1. **Lookup vs Multi-hop Boundary (Question 1):** Questions asking about an entity attribute (`Acme Corp's plan`) can be classified as `local` if the prompt treats relationship traversal as multi-hop. Local traversal succeeds, but Vector RAG is cheaper.
2. **Missing Context in Vector Retrieval (Question 4):** Question 4 requires connecting 3 facts (`Priya Nair -> Search team -> search-service -> index-db`). The router predicted `vector`, but top-2 vector chunks did not contain all facts together, resulting in "I don't know". Local traversal is required for such multi-hop queries.
3. **Entity-Heavy Global Queries (Question 6):** When a global aggregate question explicitly names specific entities (`Payments and Search teams`), the LLM classifier can mistake it for a `local` multi-hop query. Prompting the router with strict rules on dataset-wide keywords reduces this friction.

---

## 4. Honest Production Limitations Analysis

1. **Fragile Community Boundaries in Dynamic Graphs:**  
   Louvain community detection relies on graph topology at index time. Adding new nodes (e.g., new microservices or incidents) changes modularity scores, causing cluster boundaries to shift and requiring re-summarization of community reports.
2. **Ambiguous Query Routing:**  
   Borderline questions like *"How does auth-service impact Acme Corp?"* contain named entities (`auth-service`, `Acme Corp`) yet require aggregate failure impact context (`global`). Misrouting to `vector` or `local` risks missing cross-cluster dependencies.
3. **Index-Time LLM Cost & Latency at Scale:**  
   Precomputing community reports for thousands of clusters across multi-level graph hierarchies requires significant upfront LLM calls. While query-time latency is low, index-time cost scales with graph density and community count.
