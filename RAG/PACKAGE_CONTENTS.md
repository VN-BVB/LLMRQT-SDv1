# GitHub package contents

The ZIP package contains the reusable RAG source code, Markdown documentation, dependency
files, `.gitignore`, and compact experiment summaries. It intentionally excludes:

- downloaded or converted datasets under `data/`;
- FAISS, BM25, and hierarchical indexes under `storage/`;
- model checkpoints and Hugging Face caches;
- per-question predictions and other generated files under `results/`;
- Python bytecode and temporary checkpoints.

These artifacts are large, machine-specific, or reproducible from the included scripts. After
extracting the ZIP on Windows, start with `README.md`, install `requirements.txt`, and generate
the required dataset/index locally. The paths to the custom AWQ checkpoint and LLMQRT checkout
must be supplied for the target machine.

`experiment_summaries/` in the packaged archive contains only small JSON summaries used by the
README conclusions; it does not contain source documents, retrieved text, or generated answers.
