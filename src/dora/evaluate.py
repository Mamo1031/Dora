"""Evaluation script for RAG performance comparison."""

import json
import re
import sys
from pathlib import Path
from typing import Any

from dora.knowledge_base import KnowledgeBase
from dora.llm import LocalLLM


def load_questions(json_path: str | Path) -> dict[str, Any]:
    """Load evaluation questions from JSON file.

    Parameters
    ----------
    json_path : str | Path
        Path to the evaluation questions JSON file

    Returns
    -------
    dict[str, Any]
        Dictionary containing questions and metadata

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist
    """
    json_path = Path(json_path)
    if not json_path.exists():
        msg = f"Evaluation questions file not found: {json_path}"
        raise FileNotFoundError(msg)

    with json_path.open(encoding="utf-8") as f:
        data = json.load(f)

    return data["evaluation_questions"]


def normalize_text(text: str) -> str:
    """Normalize text for comparison (lowercase, remove extra whitespace).

    Parameters
    ----------
    text : str
        Text to normalize

    Returns
    -------
    str
        Normalized text
    """
    # Convert to lowercase and remove extra whitespace
    normalized = text.lower().strip()
    # Remove multiple spaces
    return re.sub(r"\s+", " ", normalized)


def extract_keywords(text: str) -> set[str]:
    """Extract keywords from text (non-stop words, numbers, important terms).

    Parameters
    ----------
    text : str
        Text to extract keywords from

    Returns
    -------
    set[str]
        Set of keywords
    """
    # Common stop words to ignore
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "should",
        "could",
        "may",
        "might",
        "must",
        "can",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "they",
        "them",
        "their",
        "what",
        "which",
        "who",
        "where",
        "when",
        "why",
        "how",
        "from",
        "as",
        "about",
        "into",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "up",
        "down",
        "out",
        "off",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
    }

    # Normalize text
    normalized = normalize_text(text)
    # Split into words
    words = re.findall(r"\b\w+\b", normalized)
    # Filter out stop words and short words (less than 2 chars), keep numbers
    return {word for word in words if (word not in stop_words and len(word) >= 2) or word.isdigit()}


def _evaluate_exact_value(answer_normalized: str, answer_keywords: set[str], expected: str) -> float:
    """Evaluate exact value type answer.

    Parameters
    ----------
    answer_normalized : str
        Normalized answer text
    answer_keywords : set[str]
        Keywords extracted from answer
    expected : str
        Expected value

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    expected_normalized = normalize_text(expected)
    expected_keywords = extract_keywords(expected)
    # Check if exact match or contains all keywords
    if expected_normalized in answer_normalized:
        return 1.0
    if expected_keywords and answer_keywords:
        matched = len(expected_keywords & answer_keywords)
        return matched / len(expected_keywords) if expected_keywords else 0.0
    return 0.0


def _evaluate_exact_values(
    answer_normalized: str,
    answer_keywords: set[str],
    expected: dict[str, str],
) -> float:
    """Evaluate exact values type answer (dict).

    Parameters
    ----------
    answer_normalized : str
        Normalized answer text
    answer_keywords : set[str]
        Keywords extracted from answer
    expected : dict[str, str]
        Expected values dictionary

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    total_score = 0.0
    total_keys = len(expected)
    for value in expected.values():
        # Look for the value in the answer
        value_normalized = normalize_text(str(value))
        value_keywords = extract_keywords(str(value))
        if value_normalized in answer_normalized:
            total_score += 1.0
        elif value_keywords and answer_keywords:
            matched = len(value_keywords & answer_keywords)
            total_score += matched / len(value_keywords) if value_keywords else 0.0
    return total_score / total_keys if total_keys > 0 else 0.0


def _evaluate_list(
    answer_normalized: str,
    answer_keywords: set[str],
    expected: list[str],
) -> float:
    """Evaluate list type answer.

    Parameters
    ----------
    answer_normalized : str
        Normalized answer text
    answer_keywords : set[str]
        Keywords extracted from answer
    expected : list[str]
        Expected list of items

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    total_score = 0.0
    total_items = len(expected)
    for item in expected:
        item_normalized = normalize_text(str(item))
        item_keywords = extract_keywords(str(item))
        # Check if item is mentioned in answer
        if item_normalized in answer_normalized:
            total_score += 1.0
        elif item_keywords and answer_keywords:
            matched = len(item_keywords & answer_keywords)
            total_score += matched / len(item_keywords) if item_keywords else 0.0
    return total_score / total_items if total_items > 0 else 0.0


def evaluate_answer(
    answer: str,
    expected: str | list[str] | dict[str, str],
    answer_type: str,
) -> float:
    """Evaluate an answer against expected answer using fuzzy matching.

    Parameters
    ----------
    answer : str
        The generated answer
    expected : str | list[str] | dict[str, str]
        Expected answer (can be string, list, or dict)
    answer_type : str
        Type of expected answer (exact_value, exact_values, list, general_knowledge)

    Returns
    -------
    float
        Score between 0.0 and 1.0
    """
    if answer_type == "general_knowledge":
        # For general knowledge questions, just check if answer is non-empty
        return 1.0 if answer.strip() else 0.0

    answer_normalized = normalize_text(answer)
    answer_keywords = extract_keywords(answer)

    evaluators: dict[str, tuple[type, Any]] = {
        "exact_value": (str, _evaluate_exact_value),
        "exact_values": (dict, _evaluate_exact_values),
        "list": (list, _evaluate_list),
    }

    if answer_type in evaluators:
        expected_type, evaluator_func = evaluators[answer_type]
        if isinstance(expected, expected_type):
            return evaluator_func(answer_normalized, answer_keywords, expected)

    return 0.0


def run_evaluation_without_rag(
    questions: list[dict[str, Any]],
    llm: LocalLLM,
) -> list[dict[str, Any]]:
    """Run evaluation without RAG (knowledge base is empty).

    Parameters
    ----------
    questions : list[dict[str, Any]]
        List of question dictionaries
    llm : LocalLLM
        LLM instance (RAG disabled)

    Returns
    -------
    list[dict[str, Any]]
        List of results, each containing question info and answer
    """
    results = []
    total = len(questions)

    for i, question in enumerate(questions, 1):
        q_id = question.get("id", i)
        q_text = question.get("question", "")
        category = question.get("category", "unknown")

        print(f"[{i}/{total}] Question {q_id} ({category}): {q_text[:60]}...")  # noqa: T201

        try:
            answer, performance = llm.invoke_with_performance(q_text)
            score = 0.0
            expected_answer = question.get("expected_answer")
            if expected_answer is not None:
                score = evaluate_answer(
                    answer,
                    expected_answer,
                    question.get("expected_answer_type", "general_knowledge"),
                )
        except RuntimeError as e:
            answer = f"Error: {e}"
            score = 0.0
            performance = {
                "total_time": 0.0,
                "generation_time": 0.0,
                "retrieval_time": 0.0,
                "ttft": 0.0,
            }

        results.append(
            {
                "question_id": q_id,
                "question": q_text,
                "category": category,
                "answer": answer,
                "score": score,
                "performance": performance,
            },
        )

    return results


def run_evaluation_with_rag(
    questions: list[dict[str, Any]],
    llm: LocalLLM,
    kb: KnowledgeBase,
    spec_pdf: Path,
) -> list[dict[str, Any]]:
    """Run evaluation with RAG (specification PDF added to knowledge base).

    Parameters
    ----------
    questions : list[dict[str, Any]]
        List of question dictionaries
    llm : LocalLLM
        LLM instance (RAG enabled)
    kb : KnowledgeBase
        Knowledge base instance
    spec_pdf : Path
        Path to specification PDF file

    Returns
    -------
    list[dict[str, Any]]
        List of results, each containing question info and answer

    Raises
    ------
    FileNotFoundError
        If specification PDF does not exist
    RuntimeError
        If document cannot be added to knowledge base
    """
    if not spec_pdf.exists():
        msg = f"Specification PDF not found: {spec_pdf}"
        raise FileNotFoundError(msg)

    print(f"Adding specification PDF to knowledge base: {spec_pdf}")  # noqa: T201
    try:
        num_chunks = kb.add_document(spec_pdf)
        print(f"✓ Added {num_chunks} chunks to knowledge base\n")  # noqa: T201
    except (FileNotFoundError, ValueError, RuntimeError) as e:
        msg = f"Failed to add specification PDF: {e}"
        raise RuntimeError(msg) from e

    results = []
    total = len(questions)

    for i, question in enumerate(questions, 1):
        q_id = question.get("id", i)
        q_text = question.get("question", "")
        category = question.get("category", "unknown")

        print(f"[{i}/{total}] Question {q_id} ({category}): {q_text[:60]}...")  # noqa: T201

        try:
            answer, performance = llm.invoke_with_performance(q_text)
            score = 0.0
            expected_answer = question.get("expected_answer")
            if expected_answer is not None:
                score = evaluate_answer(
                    answer,
                    expected_answer,
                    question.get("expected_answer_type", "general_knowledge"),
                )
        except RuntimeError as e:
            answer = f"Error: {e}"
            score = 0.0
            performance = {
                "total_time": 0.0,
                "generation_time": 0.0,
                "retrieval_time": 0.0,
                "ttft": 0.0,
            }

        results.append(
            {
                "question_id": q_id,
                "question": q_text,
                "category": category,
                "answer": answer,
                "score": score,
                "performance": performance,
            },
        )

    return results


def _calculate_score_stats(
    results_without_rag: list[dict[str, Any]],
    results_with_rag: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate score statistics for project-specific questions.

    Returns
    -------
    dict[str, float]
        Dictionary containing avg_score_without, avg_score_with, and improvement
    """
    ps_without = [r for r in results_without_rag if r["category"] == "project_specific"]
    ps_with = [r for r in results_with_rag if r["category"] == "project_specific"]

    avg_without = sum(r["score"] for r in ps_without) / len(ps_without) if ps_without else 0.0
    avg_with = sum(r["score"] for r in ps_with) / len(ps_with) if ps_with else 0.0

    return {
        "avg_score_without": avg_without,
        "avg_score_with": avg_with,
        "improvement": avg_with - avg_without,
    }


def _calculate_performance_stats(
    results_without_rag: list[dict[str, Any]],
    results_with_rag: list[dict[str, Any]],
) -> dict[str, float]:
    """Calculate performance statistics.

    Returns
    -------
    dict[str, float]
        Dictionary containing performance metrics
    """
    all_without = [r for r in results_without_rag if r.get("performance")]
    all_with = [r for r in results_with_rag if r.get("performance")]

    def avg_metric(results: list[dict[str, Any]], metric: str) -> float:
        if not results:
            return 0.0
        return sum(r["performance"].get(metric, 0) for r in results) / len(results)

    avg_ttft_without = avg_metric(all_without, "ttft")
    avg_ttft_with = avg_metric(all_with, "ttft")
    avg_total_without = avg_metric(all_without, "total_time")
    avg_total_with = avg_metric(all_with, "total_time")
    avg_retrieval = avg_metric(all_with, "retrieval_time")
    avg_gen_without = avg_metric(all_without, "generation_time")
    avg_gen_with = avg_metric(all_with, "generation_time")

    return {
        "avg_ttft_without": avg_ttft_without,
        "avg_ttft_with": avg_ttft_with,
        "avg_total_time_without": avg_total_without,
        "avg_total_time_with": avg_total_with,
        "avg_retrieval_time": avg_retrieval,
        "avg_generation_time_without": avg_gen_without,
        "avg_generation_time_with": avg_gen_with,
        "rag_overhead": avg_total_with - avg_total_without,
    }


def generate_report(
    results_without_rag: list[dict[str, Any]],
    results_with_rag: list[dict[str, Any]],
    questions_data: dict[str, Any],
) -> str:
    """Generate evaluation report in text format.

    Parameters
    ----------
    results_without_rag : list[dict[str, Any]]
        Results from evaluation without RAG
    results_with_rag : list[dict[str, Any]]
        Results from evaluation with RAG
    questions_data : dict[str, Any]
        Questions data including metadata

    Returns
    -------
    str
        Formatted report text
    """
    metadata = questions_data.get("metadata", {})
    questions = questions_data.get("questions", [])

    score_stats = _calculate_score_stats(results_without_rag, results_with_rag)
    perf_stats = _calculate_performance_stats(results_without_rag, results_with_rag)

    # Build report
    report_lines = [
        "=" * 80,
        "Dora RAG Performance Evaluation Report",
        "=" * 80,
        "",
        f"Title: {metadata.get('title', 'N/A')}",
        f"Version: {metadata.get('version', 'N/A')}",
        f"Total Questions: {metadata.get('total_questions', 0)}",
        "",
        "-" * 80,
        "EXECUTIVE SUMMARY",
        "-" * 80,
        "",
        "Project-Specific Questions (RAG Effectiveness):",
        f"  Average Score (without RAG): {score_stats['avg_score_without']:.2%}",
        f"  Average Score (with RAG):    {score_stats['avg_score_with']:.2%}",
        f"  Improvement:                 {score_stats['improvement']:+.2%}",
        "",
        "Performance Metrics:",
        "  Time to First Token (TTFT):",
        f"    Without RAG: {perf_stats['avg_ttft_without']:.3f}s",
        f"    With RAG:    {perf_stats['avg_ttft_with']:.3f}s",
        f"    Difference:  {perf_stats['avg_ttft_with'] - perf_stats['avg_ttft_without']:+.3f}s",
        "",
        "  Total Response Time (Latency):",
        f"    Without RAG: {perf_stats['avg_total_time_without']:.3f}s",
        f"    With RAG:    {perf_stats['avg_total_time_with']:.3f}s",
        "    Target:      < 10.0s",
        f"    Status:      {'✓ PASS' if perf_stats['avg_total_time_with'] < 10.0 else '✗ FAIL'}",
        "",
        "  RAG Overhead Analysis:",
        f"    Retrieval Time:        {perf_stats['avg_retrieval_time']:.3f}s",
        f"    Generation Time (no RAG): {perf_stats['avg_generation_time_without']:.3f}s",
        f"    Generation Time (RAG):     {perf_stats['avg_generation_time_with']:.3f}s",
        f"    Total RAG Overhead:    {perf_stats['rag_overhead']:.3f}s",
        "",
        "-" * 80,
        "DETAILED RESULTS",
        "-" * 80,
        "",
    ]

    # Group results by category
    for category in ["project_specific", "general_knowledge"]:
        category_name = category.replace("_", " ").title()
        report_lines.extend([
            f"{category_name} Questions:",
            "",
        ])

        for question in questions:
            if question.get("category") != category:
                continue

            q_id = question.get("id")
            # Find corresponding results
            result_without = next((r for r in results_without_rag if r["question_id"] == q_id), None)
            result_with = next((r for r in results_with_rag if r["question_id"] == q_id), None)

            if not result_without or not result_with:
                continue

            perf_without = result_without.get("performance", {})
            perf_with = result_with.get("performance", {})

            question_lines = [
                f"Question {q_id}:",
                f"  {question.get('question', 'N/A')}",
                "",
                "  Without RAG:",
                f"    Answer: {result_without['answer'][:200]}...",
            ]
            if result_without["score"] > 0:
                question_lines.append(f"    Score:  {result_without['score']:.2%}")
            question_lines.extend([
                f"    TTFT:   {perf_without.get('ttft', 0):.3f}s",
                f"    Total:  {perf_without.get('total_time', 0):.3f}s",
                "",
                "  With RAG:",
                f"    Answer: {result_with['answer'][:200]}...",
            ])
            if result_with["score"] > 0:
                question_lines.append(f"    Score:  {result_with['score']:.2%}")
            question_lines.extend([
                f"    TTFT:        {perf_with.get('ttft', 0):.3f}s",
                f"    Retrieval:   {perf_with.get('retrieval_time', 0):.3f}s",
                f"    Generation:  {perf_with.get('generation_time', 0):.3f}s",
                f"    Total:       {perf_with.get('total_time', 0):.3f}s",
            ])
            if result_with["score"] > result_without["score"]:
                improvement_q = result_with["score"] - result_without["score"]
                question_lines.append(f"    Score Improvement: +{improvement_q:.2%}")
            overhead_q = perf_with.get("total_time", 0) - perf_without.get("total_time", 0)
            if overhead_q > 0:
                question_lines.append(f"    RAG Overhead: +{overhead_q:.3f}s")
            question_lines.append("")
            if question.get("expected_answer"):
                question_lines.extend([
                    f"  Expected: {question.get('expected_answer')}",
                    "",
                ])
            question_lines.extend(["-" * 80, ""])
            report_lines.extend(question_lines)

    report_lines.extend([
        "=" * 80,
        "End of Report",
        "=" * 80,
    ])

    return "\n".join(report_lines)


def _validate_files(questions_json: Path, spec_pdf: Path) -> None:
    """Validate that required files exist."""
    if not questions_json.exists():
        print(f"✗ Error: Evaluation questions file not found: {questions_json}")  # noqa: T201
        sys.exit(1)

    if not spec_pdf.exists():
        print(f"✗ Error: Specification PDF not found: {spec_pdf}")  # noqa: T201
        print(f"  Expected location: {spec_pdf}")  # noqa: T201
        sys.exit(1)


def _load_questions_with_error_handling(questions_json: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load questions with error handling.

    Returns
    -------
    tuple[dict[str, Any], list[dict[str, Any]]]
        Tuple of (questions_data, questions list)
    """
    print(f"Loading evaluation questions from: {questions_json}")  # noqa: T201
    try:
        questions_data = load_questions(questions_json)
        questions = questions_data.get("questions", [])
        print(f"✓ Loaded {len(questions)} questions\n")  # noqa: T201
        return questions_data, questions
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"✗ Error loading questions: {e}")  # noqa: T201
        sys.exit(1)


def _initialize_llm_without_rag() -> LocalLLM:
    """Initialize LLM without RAG.

    Returns
    -------
    LocalLLM
        Initialized LLM instance without RAG
    """
    print("Initializing LLM (without RAG)...")  # noqa: T201
    try:
        llm = LocalLLM(model_name="llama3.2", use_rag=False)
        print("✓ LLM initialized\n")  # noqa: T201
        return llm
    except ConnectionError as e:
        print(f"✗ Error: {e}")  # noqa: T201
        print("\nPlease ensure:")  # noqa: T201
        print("  1. Ollama is installed and running")  # noqa: T201
        print("  2. Llama 3.2 model is pulled: ollama pull llama3.2")  # noqa: T201
        sys.exit(1)


def _run_phase1(questions: list[dict[str, Any]], llm: LocalLLM) -> tuple[list[dict[str, Any]], KnowledgeBase]:
    """Run Phase 1: Evaluation without RAG.

    Returns
    -------
    tuple[list[dict[str, Any]], KnowledgeBase]
        Tuple of (evaluation results, knowledge base instance)
    """
    print("=" * 80)  # noqa: T201
    print("Phase 1: Evaluation WITHOUT RAG")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    kb = KnowledgeBase()
    kb_info = kb.get_info()
    if kb_info["exists"] and kb_info["count"] > 0:
        print(f"Clearing knowledge base ({kb_info['count']} chunks)...")  # noqa: T201
        kb.clear()
        print("✓ Knowledge base cleared\n")  # noqa: T201

    results = run_evaluation_without_rag(questions, llm)
    print("\n✓ Phase 1 completed\n")  # noqa: T201
    return results, kb


def _run_phase2(
    questions: list[dict[str, Any]],
    kb: KnowledgeBase,
    spec_pdf: Path,
) -> list[dict[str, Any]]:
    """Run Phase 2: Evaluation with RAG.

    Returns
    -------
    list[dict[str, Any]]
        Evaluation results with RAG
    """
    print("=" * 80)  # noqa: T201
    print("Phase 2: Evaluation WITH RAG")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    try:
        llm_with_rag = LocalLLM(model_name="llama3.2", use_rag=True, knowledge_base=kb)
        print("✓ LLM with RAG initialized\n")  # noqa: T201
    except ConnectionError as e:
        print(f"✗ Error: {e}")  # noqa: T201
        sys.exit(1)

    try:
        results = run_evaluation_with_rag(questions, llm_with_rag, kb, spec_pdf)
        print("\n✓ Phase 2 completed\n")  # noqa: T201
        return results
    except (FileNotFoundError, RuntimeError) as e:
        print(f"✗ Error: {e}")  # noqa: T201
        sys.exit(1)


def _save_report(report: str, output_path: Path) -> None:
    """Save report to file."""
    print("=" * 80)  # noqa: T201
    print("Generating Report")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    try:
        output_path.write_text(report, encoding="utf-8")
        print(f"✓ Report saved to: {output_path}")  # noqa: T201
    except OSError as e:
        print(f"✗ Error saving report: {e}")  # noqa: T201
        sys.exit(1)


def _print_summary(
    results_without_rag: list[dict[str, Any]],
    results_with_rag: list[dict[str, Any]],
    output_report: Path,
) -> None:
    """Print evaluation summary."""
    print()  # noqa: T201
    print("=" * 80)  # noqa: T201
    print("Evaluation Complete")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    ps_without = [r for r in results_without_rag if r["category"] == "project_specific"]
    ps_with = [r for r in results_with_rag if r["category"] == "project_specific"]

    all_without = [r for r in results_without_rag if r.get("performance")]
    all_with = [r for r in results_with_rag if r.get("performance")]

    if ps_without and ps_with:
        avg_without = sum(r["score"] for r in ps_without) / len(ps_without)
        avg_with = sum(r["score"] for r in ps_with) / len(ps_with)
        improvement = avg_with - avg_without

        print("Summary:")  # noqa: T201
        print(f"  Project-Specific Questions Average Score (without RAG): {avg_without:.2%}")  # noqa: T201
        print(f"  Project-Specific Questions Average Score (with RAG):    {avg_with:.2%}")  # noqa: T201
        print(f"  Improvement:                                            {improvement:+.2%}")  # noqa: T201

    if all_without and all_with:
        avg_ttft_without = sum(r["performance"].get("ttft", 0) for r in all_without) / len(all_without)
        avg_ttft_with = sum(r["performance"].get("ttft", 0) for r in all_with) / len(all_with)
        avg_total_without = sum(r["performance"].get("total_time", 0) for r in all_without) / len(all_without)
        avg_total_with = sum(r["performance"].get("total_time", 0) for r in all_with) / len(all_with)
        avg_retrieval = sum(r["performance"].get("retrieval_time", 0) for r in all_with) / len(all_with)
        rag_overhead = avg_total_with - avg_total_without

        print("\nPerformance Summary:")  # noqa: T201
        print(f"  Average TTFT (without RAG):     {avg_ttft_without:.3f}s")  # noqa: T201
        print(f"  Average TTFT (with RAG):        {avg_ttft_with:.3f}s")  # noqa: T201
        print(f"  Average Total Time (without RAG): {avg_total_without:.3f}s")  # noqa: T201
        print(f"  Average Total Time (with RAG):    {avg_total_with:.3f}s")  # noqa: T201
        print(f"  Target (< 10s):                   {'✓ PASS' if avg_total_with < 10.0 else '✗ FAIL'}")  # noqa: T201
        print(f"  Average Retrieval Time:          {avg_retrieval:.3f}s")  # noqa: T201
        print(f"  RAG Overhead:                    {rag_overhead:+.3f}s")  # noqa: T201

    print(f"\nFull report available at: {output_report}")  # noqa: T201


def main() -> None:
    """Run evaluation script to compare RAG performance."""
    project_root = Path(__file__).parent.parent.parent
    evaluation_dir = project_root / "evaluation"
    questions_json = evaluation_dir / "questions.json"
    spec_pdf = evaluation_dir / "specification.pdf"
    output_report = evaluation_dir / "reports" / "evaluation_report.txt"

    _validate_files(questions_json, spec_pdf)

    print("=" * 80)  # noqa: T201
    print("Dora RAG Performance Evaluation")  # noqa: T201
    print("=" * 80)  # noqa: T201
    print()  # noqa: T201

    questions_data, questions = _load_questions_with_error_handling(questions_json)
    llm_without_rag = _initialize_llm_without_rag()
    results_without_rag, kb = _run_phase1(questions, llm_without_rag)
    results_with_rag = _run_phase2(questions, kb, spec_pdf)

    report = generate_report(results_without_rag, results_with_rag, questions_data)
    _save_report(report, output_report)
    _print_summary(results_without_rag, results_with_rag, output_report)
