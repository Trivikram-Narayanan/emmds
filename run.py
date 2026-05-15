"""
EMMDS — Entry Point
====================
Run the full pipeline from the command line, or launch the UI/API.

Usage:
  python run.py --test          # Demo on Iris dataset
  python run.py --ui            # Launch Streamlit UI (localhost:8501)
  python run.py --api           # Launch FastAPI backend (localhost:8000)
  python run.py --all           # Launch both simultaneously
  python run.py --csv data.csv --target label   # Your own CSV
  python run.py --benchmark     # Run experiments on 56 datasets
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))


def check_deps():
    """Check optional dependencies and warn if missing."""
    optional = {
        'fastapi':   'FastAPI backend  (pip install fastapi uvicorn)',
        'streamlit': 'Streamlit UI     (pip install streamlit plotly)',
        'shap':      'SHAP explainer   (pip install shap)',
        'lime':      'LIME explainer   (pip install lime)',
        'xgboost':   'XGBoost model    (pip install xgboost)',
        'lightgbm':  'LightGBM model   (pip install lightgbm)',
        'openml':    'OpenML datasets  (pip install openml)',
        'ctgan':     'CTGAN augment    (pip install ctgan)',
    }
    missing = []
    for pkg, desc in optional.items():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(f"  ⚠  {desc}")

    if missing:
        print("Optional packages not installed:")
        for m in missing:
            print(m)
        print("Core ML pipeline works without these.\n")


def run_test():
    """Run demo on Iris dataset."""
    from sklearn.datasets import load_iris
    import pandas as pd
    from src.pipeline.orchestrator import PipelineOrchestrator

    print("Running EMMDS demo on Iris dataset...")
    data = load_iris(as_frame=True)
    df = data.frame.copy(); df["target"] = data.target

    orch = PipelineOrchestrator()
    orch.run_from_dataframe(df=df, target_col="target")
    orch.print_summary()


def run_pipeline(csv_path, target_col, task=None, scaler="standard"):
    """Run pipeline on a CSV file."""
    from src.pipeline.orchestrator import PipelineOrchestrator
    orch = PipelineOrchestrator()
    result = orch.run_from_file(
        csv_path=csv_path, target_col=target_col,
        task=task, scaler=scaler, save_results=True,
    )
    orch.print_summary()
    return result


def run_benchmark():
    """Run experiments on full dataset collection."""
    from src.research.benchmark import BenchmarkEngine
    bench = BenchmarkEngine()
    bench.add_sklearn_dataset("breast_cancer")
    bench.add_sklearn_dataset("wine")
    bench.add_sklearn_dataset("iris")
    report = bench.run()
    bench.print_summary()
    bench.save_report(report)


def launch_api(host="0.0.0.0", port=8000):
    """Launch FastAPI backend."""
    try:
        import uvicorn
    except ImportError:
        print("❌ FastAPI not installed. Run: pip install fastapi uvicorn")
        return
    print(f"\n🚀 Starting EMMDS API at http://{host}:{port}")
    print(f"   Docs: http://{host}:{port}/docs\n")
    uvicorn.run("api.main:app", host=host, port=port, reload=True)


def launch_ui():
    """Launch Streamlit frontend."""
    try:
        import streamlit
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit plotly")
        return
    import subprocess
    print("\n🎨 Starting EMMDS Streamlit UI...")
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        str(ROOT / "app" / "app.py"),
        "--server.port", "8501",
        "--server.address", "localhost",
    ])


def launch_all():
    """Launch both API and UI in parallel."""
    import threading
    t = threading.Thread(target=launch_api, daemon=True)
    t.start()
    launch_ui()


def main():
    parser = argparse.ArgumentParser(
        description="EMMDS — Ensemble Multi-Model Decision System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--csv",       type=str,  help="Path to CSV file")
    parser.add_argument("--target",    type=str,  help="Target column name")
    parser.add_argument("--task",      type=str,  default=None,
                        choices=["classification", "regression"])
    parser.add_argument("--scaler",    type=str,  default="standard",
                        choices=["standard", "minmax", "none"])
    parser.add_argument("--api",       action="store_true", help="Launch FastAPI")
    parser.add_argument("--ui",        action="store_true", help="Launch Streamlit")
    parser.add_argument("--all",       action="store_true", help="Launch both")
    parser.add_argument("--test",      action="store_true", help="Demo on Iris")
    parser.add_argument("--benchmark", action="store_true", help="Run benchmark")

    args = parser.parse_args()
    check_deps()

    if args.all:
        launch_all()
    elif args.api:
        launch_api()
    elif args.ui:
        launch_ui()
    elif args.benchmark:
        run_benchmark()
    elif args.test:
        run_test()
    elif args.csv and args.target:
        run_pipeline(args.csv, args.target, args.task, args.scaler)
    else:
        parser.print_help()
        print("\n💡 Quick start:")
        print("   python run.py --test           # Demo on Iris")
        print("   python run.py --ui             # Streamlit UI")
        print("   python run.py --api            # FastAPI backend")
        print("   python run.py --all            # Both")
        print("   python run.py --benchmark      # Run experiments")
        print("   python run.py --csv data.csv --target label")


if __name__ == "__main__":
    main()
