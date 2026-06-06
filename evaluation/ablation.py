"""Ablation study runner."""
from dataclasses import dataclass, field
from typing import Dict, List, Callable


@dataclass
class AblationConfig:
    name: str
    components: Dict[str, bool] = field(default_factory=dict)
    description: str = ""


@dataclass
class AblationResult:
    config_name: str
    metrics: Dict[str, float] = field(default_factory=dict)
    fps: float = 0.0


class AblationStudy:
    def __init__(self):
        self._configs: List[AblationConfig] = []
        self._eval_fn: Callable = None

    def set_eval_function(self, fn: Callable):
        self._eval_fn = fn

    def add_config(self, name: str, components: Dict[str, bool], description: str = ""):
        self._configs.append(AblationConfig(name, components, description))

    def add_defaults(self):
        full = {"pose3d": True, "quaternion": True, "sparc": True, "compensation": True, "fatigue": True, "llm": True}
        self.add_config("full_system", full, "Full system")
        for comp in full:
            cfg = full.copy()
            cfg[comp] = False
            self.add_config(f"no_{comp}", cfg, f"Without {comp}")

    def run(self, test_data, ground_truth) -> List[AblationResult]:
        if not self._eval_fn:
            raise ValueError("Set eval function first")
        return [AblationResult(c.name, self._eval_fn(c, test_data, ground_truth)) for c in self._configs]

    def print_report(self, results: List[AblationResult]):
        if not results: return
        metrics = list(results[0].metrics.keys())
        print(f"\n{'Config':<20} " + " ".join(f"{m:>12}" for m in metrics))
        print("-" * (20 + 13 * len(metrics)))
        for r in results:
            print(f"{r.config_name:<20} " + " ".join(f"{r.metrics.get(m, 0):>12.3f}" for m in metrics))
