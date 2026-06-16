"""
main.py — Member 4 CLI entry point
Hidden Variables RAG Project

Quick demo / sanity-check of the explainability engine.
Run:  python main.py
"""

from explainability import ExplainabilityEngine


def print_separator(char="─", width=60):
    print(char * width)


def main():
    print_separator("═")
    print("  Hidden Variables · Member 4: Explainability & Evaluation")
    print_separator("═")

    engine = ExplainabilityEngine()
    chunk_count = engine.vector_store._collection.count()
    print(f"  Vector store loaded — {chunk_count} chunks indexed\n")

    if chunk_count == 0:
        print("  ⚠️  No vectors found.  Run Sneha's ingestion pipeline first:")
        print("       python ../Hidden-Variables-sneha/main.py")
        print()

    while True:
        print_separator()
        user_input = input("  Question (q to quit, 's' to summarise a PDF): ").strip()

        if user_input.lower() == "q":
            break

        if user_input.lower() == "s":
            pdf_path = input("  PDF path: ").strip()
            summary = engine.summarize_pdf(pdf_path)
            print(f"\n  📋 Executive Summary\n  {summary.executive_summary}")
            print(f"\n  🏷️  Key Topics")
            for t in summary.key_topics:
                print(f"    • {t}")
            print(f"\n  🎯 Main Conclusions")
            for c in summary.main_conclusions:
                print(f"    • {c}")
            continue

        if not user_input:
            continue

        result = engine.ask(user_input)

        print()
        if not result.found_in_document:
            print("  ❌ Answer not found in document.")
        else:
            print(f"  ✅ Answer: {result.answer}")
            print(f"  📊 Confidence: {result.top_confidence}%  |  ⏱️  {result.latency_ms} ms")
            print()
            print("  📌 Sources:")
            for i, src in enumerate(result.sources, 1):
                print(
                    f"    [{i}] {src.filename} — "
                    f"Page {src.page}, Paragraph {src.paragraph} "
                    f"(conf: {src.confidence_pct}%)"
                )
                print(f"        \"{src.snippet}…\"")
        print()

    # Print session metrics on exit
    print_separator("═")
    print("  Session Metrics")
    print_separator()
    m = engine.get_metrics()
    for k, v in m.items():
        label = k.replace("_", " ").title()
        print(f"  {label:<30} {v}")
    print_separator("═")


if __name__ == "__main__":
    main()
