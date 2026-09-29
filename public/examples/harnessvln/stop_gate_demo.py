"""Educational stop gate inspired by HarnessVLN, not the authors' source code.

Run with Python 3.10+: python3 stop_gate_demo.py
This is not a navigation controller or a safety-certified component.
Appendix 1 supplies the 2.5 m / 1.0 m / 0.5 m thresholds. Evidence freshness
is represented as an input predicate; no timestamp or pose estimator is supplied.
The checks below exercise illustrative inputs, not Habitat episodes or hardware.
"""

from dataclasses import dataclass, replace
from math import isfinite


@dataclass(frozen=True)
class Evidence:
    semantic_match: bool
    endpoint_verified: bool
    fresh_and_pose_aligned: bool
    depth_m: float | None
    depth_std_m: float | None
    projected_target_distance_m: float | None


def nonnegative_finite(value: float | None) -> bool:
    return value is not None and isfinite(value) and value >= 0


def geometric_gate(e: Evidence) -> tuple[bool, str]:
    reliable_depth = (
        nonnegative_finite(e.depth_m)
        and e.depth_m > 0
        and nonnegative_finite(e.depth_std_m)
        and e.depth_std_m <= 0.5
    )
    if reliable_depth:
        # A reliable far-depth measurement cannot be overridden by a close
        # projected waypoint. Fallback is only for unavailable/unreliable depth.
        return e.depth_m <= 2.5, "fresh_depth"
    if nonnegative_finite(e.projected_target_distance_m):
        return e.projected_target_distance_m <= 1.0, "projected_target"
    return False, "no_valid_distance"


def request_stop(e: Evidence) -> tuple[bool, str]:
    if not e.fresh_and_pose_aligned:
        return False, "reobserve"
    if not e.semantic_match:
        return False, "refine_target"
    if not e.endpoint_verified:
        return False, "verify_task_endpoint"
    return geometric_gate(e)


def run_examples() -> None:
    ready = Evidence(True, True, True, 1.2, 0.1, 0.4)
    cases = [
        ("near target", ready, (True, "fresh_depth")),
        ("fresh depth overrides a close waypoint",
         replace(ready, depth_m=4.65), (False, "fresh_depth")),
        ("no reliable depth: use projected target",
         replace(ready, depth_std_m=0.8), (True, "projected_target")),
        ("stale or mismatched observation",
         replace(ready, fresh_and_pose_aligned=False), (False, "reobserve")),
        ("wrong semantic identity",
         replace(ready, semantic_match=False), (False, "refine_target")),
        ("task endpoint not established",
         replace(ready, endpoint_verified=False), (False, "verify_task_endpoint")),
        ("invalid numbers are not near distances",
         replace(ready, depth_m=float("nan"),
                 projected_target_distance_m=float("nan")),
         (False, "no_valid_distance")),
        ("missing measurements",
         replace(ready, depth_m=None, projected_target_distance_m=None),
         (False, "no_valid_distance")),
    ]
    for label, evidence, expected in cases:
        actual = request_stop(evidence)
        if actual != expected:
            raise AssertionError((label, actual, expected))
        print(f"{label}: {actual}")
    # A successful internal gate does not establish benchmark success. In the
    # paper's OVON 2469 example, 1.20 m depth passes but the evaluator reports
    # 2.15 m to its goal region. Evaluator ground truth is NOT a gate input.
    print("Gate acceptance is a decision, not proof of ground-truth task success.")


if __name__ == "__main__":
    run_examples()
