import csv
import pickle
from pathlib import Path


def export_model_metrics(model_name: str, metrics_path: Path, out_path: Path) -> None:
    with metrics_path.open("rb") as f:
        metrics = pickle.load(f)

    rows = []
    for slot_info in metrics["per_slot"]:
        slot = int(slot_info["slot"])
        slot_type = slot_info["type"]
        for var in ("KL", "KR", "KLR"):
            vm = slot_info["per_variable"][var]
            rows.append(
                {
                    "model": model_name,
                    "slot": slot,
                    "slot_type": slot_type,
                    "variable": var,
                    "rmse": float(vm["rmse"]),
                    "mae": float(vm["mae"]),
                    "r2": float(vm["r2"]),
                }
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="ascii") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["model", "slot", "slot_type", "variable", "rmse", "mae", "r2"],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent

    jobs = [
        ("M6", root / "M6" / "model_metrics.pkl", root / "M6" / "slot_variable_metrics_m6.csv"),
        ("M5", root / "M5" / "model_metrics.pkl", root / "M6" / "slot_variable_metrics_m5.csv"),
    ]

    for model_name, metrics_path, out_path in jobs:
        export_model_metrics(model_name, metrics_path, out_path)
        print(f"Exported {model_name} -> {out_path}")
