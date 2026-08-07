#!/usr/bin/env python3
r"""
trace_modules.py

A tracing script that measures which Python modules are actually used and
should therefore be included in the redistributable package.

Run the real API calls used by your product inside run_traced_workload()
(such as MFR inference, shape embedding generation, and similarity-search
indexing), then extract the difference in top-level package names added to
sys.modules before and after the workload runs.

Usage:
    export HOOPS_AI_HOME=/path/to/HOOPS_AI/V1.1
    export HOOPS_AI_LICENSE='...'
    export DISPLAY=:99   # In headless environments, make sure Xvfb is already running
    python3 tools/trace_modules.py \
        --cad-file /path/to/sample.step \
        --mfr-ckpt /path/to/ts3d_162k_mfr.ckpt \
        --embed-ckpt /path/to/ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt \
        --out /tmp/used_top_level_modules.txt

The site-packages path is built automatically from $HOOPS_AI_HOME
(Linux: $HOOPS_AI_HOME/.venv/lib/python3.12/site-packages,
 Windows: %HOOPS_AI_HOME%\.venv\Lib\site-packages).

If the features used by your product are different from MFR inference, shape
embedding generation, and similarity-search indexing (for example, if it only
converts CAD files), rewrite the contents of run_traced_workload() to call the
actual APIs you use and run it again. If you add or change the invoked
features, you must rerun this trace and update the contents of python_packages/
accordingly. For example, when similarity-search index functionality was added
to hoops_ai_native_bridge (such as HoopsAI_AddCADToIndex, which internally uses
hoops_ai.ml.embeddings.FaissVectorStore), this script was also updated to add a
call through FaissVectorStore and the trace was rerun. Likewise, when thumbnail
generation was added (HoopsAI_AddCADToIndex now renders a PNG through
hoops_ai.cadaccess.HOOPSTools.exportStreamCache), a call to exportStreamCache was
added here and the trace was rerun.
"""
import argparse
import os
import sys


STDLIB_MODULE_NAMES = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()


def resolve_site_packages() -> str:
    """Build the venv site-packages path for a standard HOOPS AI installation
    from HOOPS_AI_HOME."""
    home = os.environ.get("HOOPS_AI_HOME", "")
    if not home:
        return ""

    if os.name == "nt":
        return os.path.join(home, ".venv", "Lib", "site-packages")
    return os.path.join(home, ".venv", "lib", "python3.12", "site-packages")



def top_level_name(module_name: str) -> str:
    return module_name.split(".")[0]


def run_traced_workload(cad_file: str, mfr_ckpt: str, embed_ckpt: str) -> None:
    """Call the features actually used by your product through the real HOOPS AI APIs.
    Rewrite this function to match your product's real use case.
    (The calls in this sample were verified with hoops_ai 1.1.0. If the version
    changes, the import paths or arguments may also change, so verify them.)"""
    import hoops_ai  # noqa: F401

    license_key = os.environ.get("HOOPS_AI_LICENSE", "")
    if license_key:
        hoops_ai.set_license(license_key, validate=True)

    from hoops_ai.cadaccess import HOOPSLoader

    loader = HOOPSLoader()

    # --- MFR inference (per-face feature classification) ---
    if mfr_ckpt:
        from hoops_ai.ml.EXPERIMENTAL import FlowInference, GraphNodeClassification

        flow_model = GraphNodeClassification(result_dir=".")
        model = FlowInference(cad_loader=loader, flowmodel=flow_model)
        model.load_from_checkpoint(mfr_ckpt)
        ml_input = model.preprocess(cad_file)
        model.predict_and_postprocess(ml_input)

    # --- Shape embedding generation ---
    if embed_ckpt:
        from hoops_ai.ml.embeddings import HOOPSEmbeddings

        HOOPSEmbeddings.register_model(model_name="traced_model", checkpoint_path=embed_ckpt)
        embedder = HOOPSEmbeddings(cad_loader=loader, model="traced_model")
        raw_embeddings = embedder.embed_shape(cad_file)

        # --- Similarity-search index (hoops_ai.ml.embeddings.FaissVectorStore used
        #     internally by hoops_ai_native_bridge functions such as
        #     HoopsAI_AddCADToIndex / HoopsAI_SearchIndex) ---
        import tempfile

        import numpy as np
        from hoops_ai.ml.embeddings import Embedding, FaissVectorStore, VectorRecord

        vectors = [np.asarray(e.values, dtype="float32") for e in raw_embeddings]
        vec = vectors[0] if len(vectors) == 1 else np.mean(vectors, axis=0)
        dim = int(vec.shape[0])

        with tempfile.TemporaryDirectory() as tmp_dir:
            index_base = os.path.join(tmp_dir, "traced_index")

            vs = FaissVectorStore(dim)
            emb = Embedding(values=vec, model="traced_model", dim=dim)
            rec = VectorRecord(id=cad_file, embedding=emb, metadata={"file_id": cad_file})
            vs.upsert([rec])
            vs.save(index_base)

            vs = FaissVectorStore.load(index_base)
            vs.query(vec, top_k=1)

        # --- Thumbnail generation (hoops_ai.cadaccess.HOOPSTools.exportStreamCache,
        #     used internally by hoops_ai_native_bridge HoopsAI_AddCADToIndex to render
        #     <indexDir>/thumbnails/<stem>.png). This must be exercised here so the
        #     redistributable package captures the PNG/stream-cache export modules. ---
        from hoops_ai.cadaccess import HOOPSTools

        with tempfile.TemporaryDirectory() as thumb_dir:
            model = loader.create_from_file(cad_file)
            tools = HOOPSTools()
            tools.exportStreamCache(
                model,
                filename=os.path.join(thumb_dir, "traced_thumb.scs"),
                is_white_background=True,
                overwrite=True,
            )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cad-file", required=True)
    parser.add_argument("--mfr-ckpt", default="")
    parser.add_argument("--embed-ckpt", default="")
    parser.add_argument("--out", default="/tmp/used_top_level_modules.txt")
    args = parser.parse_args()

    site_packages = resolve_site_packages()
    if site_packages:
        sys.path.insert(0, site_packages)

    before = set(sys.modules.keys())
    run_traced_workload(args.cad_file, args.mfr_ckpt, args.embed_ckpt)
    after = set(sys.modules.keys())

    new_modules = after - before
    top_level = sorted({top_level_name(m) for m in new_modules if top_level_name(m) not in STDLIB_MODULE_NAMES})

    with open(args.out, "w") as f:
        for name in top_level:
            f.write(name + "\n")

    print(f"Newly imported top-level packages: {len(top_level)}")
    print(f"Wrote package list to: {args.out}")
    print("(Use this list to pass only the required directories/files from")
    print(" site-packages to tools/build_redist_package.sh)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
