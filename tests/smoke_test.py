"""End-to-end smoke test: runs every page headlessly with Streamlit's AppTest.

Usage (from the project folder, with the dataset in data/):
    python tests/smoke_test.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from streamlit.testing.v1 import AppTest  # noqa: E402

PAGES = ["views/command_center.py", "views/product_intelligence.py", "views/price_predictor.py",
         "views/market_analytics.py", "views/product_comparison.py", "views/ai_insights.py",
         "views/model_lab.py", "views/about.py"]
TIMEOUT = 600  # the first run trains the models


def run(path: str, mutate=None) -> AppTest:
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=TIMEOUT)
    at.run()
    at.switch_page(path)
    at.run()
    if mutate:
        mutate(at)
        at.run()
    return at


def check(name: str, at: AppTest) -> bool:
    problems = [e.value for e in at.exception] + [e.value for e in at.error]
    status = "ok" if not problems else "FAILED: " + " | ".join(str(p)[:200] for p in problems)
    print(f"{name:<55} {status}")
    return not problems


def main() -> int:
    passed = True
    for page in PAGES:
        passed &= check(page, run(page))

    def unseen_inputs(at):
        at.selectbox(key="pp_brand").set_value("✎ Enter a new value…")
        at.run()
        at.text_input(key="pp_brand_new").set_value("Totally New Brand Ltd")
        at.number_input(key="pp_mrp").set_value(250000.0)
    at = run("views/price_predictor.py", unseen_inputs)
    passed &= check("predictor: unseen brand + out-of-range MRP", at)
    passed &= bool(at.warning)

    def no_matches(at):
        at.text_input[0].set_value("zzzz-no-such-product-zzzz")
    passed &= check("product intelligence: search with no matches", run("views/product_intelligence.py", no_matches))

    def compare_three(at):
        import pandas as pd
        from src.data_loader import load_dataset
        df, _ = load_dataset()
        sub = df["Subcategory"].value_counts().index[0]
        at.selectbox[0].set_value(sub)
        at.run()
        at.button[0].click()
    passed &= check("comparison: quick-pick three products", run("views/product_comparison.py", compare_three))

    print("\nALL PASSED" if passed else "\nSOME CHECKS FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
