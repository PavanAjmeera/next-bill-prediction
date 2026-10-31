"""Run every layer in order:  python run_pipeline.py"""
import time

from src.data import clean, generate
from src.features import build
from src.model import evaluate, train

STEPS = [("1. Generate data", generate.main), ("2. Clean and validate", clean.main),
         ("3. Build features", build.main), ("4. Train and predict", train.main),
         ("5. Evaluate, explain, flag", evaluate.main)]

if __name__ == "__main__":
    for name, step in STEPS:
        start = time.time()
        print(f"\n=== {name} ===")
        step()
        print(f"({time.time() - start:.1f}s)")
    print("\nDone. Start the dashboard with:  streamlit run app/dashboard.py")
