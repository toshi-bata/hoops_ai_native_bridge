# repro_plain_python.py
# Purpose: Reproduce the same call without going through the C++ embedding,
#          using the venv's python.exe directly, to find the real cause
#          (full traceback) of "construct loader/flowmodel failed: ...
#          returned a result with an exception set".
#
# How to run (from PowerShell):
#   C:\SDK\HOOPS_AI\V1.1\.venv\Scripts\python.exe repro_plain_python.py

import traceback

try:
    import hoops_ai
    # Get the license key from the HOOPS_AI_LICENSE environment variable
    # (try with an empty string if not set)
    import os
    key = os.environ.get("HOOPS_AI_LICENSE", "")
    if key:
        hoops_ai.set_license(key, validate=True)

    from hoops_ai.cadaccess import HOOPSLoader
    from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphNodeClassification

    loader = HOOPSLoader()

    # Exactly the same call as on the C++ side: only result_dir is specified
    try:
        flow_model = GraphNodeClassification(result_dir=".")
        print("GraphNodeClassification(result_dir='.') OK:", flow_model)
    except Exception:
        print("=== Attempt 1 failed (result_dir only) ===")
        traceback.print_exc()

        # The GraphClassification call in notebook 4c passed num_classes.
        # Suspecting GraphNodeClassification may have a similar required
        # argument, retry passing MFR's 25 classes (0=none, 1..24=feature types).
        print("\n=== Attempt 2: also specifying num_classes=25 ===")
        flow_model = GraphNodeClassification(num_classes=25, result_dir=".")
        print("GraphNodeClassification(num_classes=25, result_dir='.') OK:", flow_model)

except Exception:
    print("=== FULL TRACEBACK ===")
    traceback.print_exc()

    # SystemError etc. can hide the real cause in __context__/__cause__, so trace it
    import sys
    exc = sys.exc_info()[1]
    depth = 0
    while exc is not None and depth < 5:
        print(f"--- chained exception (depth {depth}): {type(exc).__name__}: {exc} ---")
        exc = getattr(exc, "__context__", None) or getattr(exc, "__cause__", None)
        depth += 1
