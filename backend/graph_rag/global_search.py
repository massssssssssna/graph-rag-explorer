"""
backend/graph_rag/global_search.py
Executes Global Search using Map-Reduce over precomputed community reports.
Map: Evaluates question against each community report independently.
Reduce: Aggregates partial answers into a comprehensive global conclusion.
"""
import logging
from backend.llm import groq_client
from backend.graph_rag.communities import community_manager

logger = logging.getLogger(__name__)

MAP_PROMPT = """You are analyzing a cluster report for a global question: "{question}".
If this report mentions any services, failures, causes, dependencies, or customers relevant to the question, summarize those key facts in 1-2 bullet points.
If completely irrelevant, reply ONLY with "NOTHING".

Cluster Report:
{report}"""

REDUCE_PROMPT = """You are answering the question: "{question}"

Combine all the following findings extracted from different clusters of the dataset into a single direct, definitive answer.
Name the primary root cause(s) and patterns that appear across multiple clusters.

Findings from Cluster Reports:
{partials}

Answer:"""


def global_search(question: str, kg_instance=None) -> dict:
    """
    Map-Reduce Global Search pipeline.
    Returns final answer, partial findings, and community reports used.
    """
    reports = community_manager.reports
    if not reports and kg_instance is not None:
        community_manager.detect_communities(kg_instance)
        reports = community_manager.generate_reports(kg_instance)

    if not reports:
        return {
            "answer": "No community reports available to perform global search.",
            "map_partials": [],
            "reports_used": 0
        }

    partials = []
    map_details = []

    # Map Phase
    for i, report in enumerate(reports):
        try:
            map_res = groq_client.chat(
                system_prompt="You evaluate cluster reports for global Q&A map step.",
                user_prompt=MAP_PROMPT.format(report=report, question=question),
                temperature=0.0,
            )
            if "NOTHING" not in map_res.strip().upper():
                partials.append(map_res)
                map_details.append({"community_id": i, "finding": map_res})
        except Exception as exc:
            logger.error("Error during Map step for community %d: %s", i, exc)

    if not partials:
        return {
            "answer": "Global search found no relevant insights across the community reports for this question.",
            "map_partials": [],
            "reports_used": len(reports)
        }

    # Reduce Phase
    joined_partials = "\n".join(f"- Community Finding: {p}" for p in partials)
    try:
        final_answer = groq_client.chat(
            system_prompt="You synthesize partial findings into a final global answer.",
            user_prompt=REDUCE_PROMPT.format(question=question, partials=joined_partials),
            temperature=0.0,
        )
    except Exception as exc:
        logger.error("Error during Reduce step: %s", exc)
        final_answer = f"Reduce synthesis failed: {exc}"

    return {
        "answer": final_answer,
        "map_partials": map_details,
        "reports_used": len(reports)
    }
