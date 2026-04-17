#!/usr/bin/env python3
"""
Bulk ingestion script for Halcyon Ridge creative corpus into MemOS.
Processes PDF, DOCX, MD, TXT, HTML files — skips images.

Reads each file as plain text locally, then passes the content directly
to mos.add(memory_content=...) to bypass tree_text scene extraction,
which requires a larger LLM than qwen2.5:0.5b.

Usage:
    python D:\MemOS\ingest_halcyon.py
"""

import os
import sys
import logging
import warnings

# Silence noisy output
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_ROOT = os.getenv("CORPUS_ROOT",           r"D:\Halcyon Ridge")
OLLAMA_BASE = os.getenv("OLLAMA_API_BASE",       "http://100.72.225.62:11434")
CHAT_MODEL  = os.getenv("OLLAMA_CHAT_MODEL",     "qwen2.5:0.5b")
EMBED_MODEL = os.getenv("OLLAMA_EMBEDDER_MODEL", "nomic-embed-text:latest")
EMBED_DIM   = int(os.getenv("EMBEDDING_DIMENSION", "1024"))
NEO4J_URI   = os.getenv("NEO4J_URI",             "bolt://100.72.225.62:7687")
NEO4J_USER  = os.getenv("NEO4J_USER",            "neo4j")
NEO4J_PASS  = os.getenv("NEO4J_PASSWORD",        "EchoZulu11!!")
NEO4J_DB    = os.getenv("NEO4J_DB_NAME",         "neo4j")
USER_ID     = os.getenv("MOS_USER_ID",           "claude_user")

SUPPORTED_EXT = {".pdf", ".docx", ".md", ".txt", ".html"}


# ── Text extraction ───────────────────────────────────────────────────────────
def extract_text(filepath: str) -> str:
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".pdf":
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            return "\n\n".join(
                page.extract_text() or "" for page in reader.pages
            ).strip()
        except ImportError:
            raise RuntimeError("pypdf not installed. Run: pip install pypdf")

    if ext == ".docx":
        try:
            import docx
            doc = docx.Document(filepath)
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except ImportError:
            raise RuntimeError("python-docx not installed. Run: pip install python-docx")

    if ext in (".md", ".txt"):
        with open(filepath, encoding="utf-8", errors="replace") as f:
            return f.read().strip()

    if ext == ".html":
        try:
            from bs4 import BeautifulSoup
            with open(filepath, encoding="utf-8", errors="replace") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
            return soup.get_text(separator="\n").strip()
        except ImportError:
            # Fallback: strip tags with regex
            import re
            with open(filepath, encoding="utf-8", errors="replace") as f:
                raw = f.read()
            return re.sub(r"<[^>]+>", " ", raw).strip()

    return ""


# ── Build MOS instance ────────────────────────────────────────────────────────
def build_mos():
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.configs.mem_os import MOSConfig
    from memos.mem_cube.general import GeneralMemCube
    from memos.mem_os.main import MOS

    ollama_llm = {
        "backend": "ollama",
        "config": {
            "model_name_or_path": CHAT_MODEL,
            "api_base": OLLAMA_BASE,
            "temperature": 0.7,
            "max_tokens": 2048,
            "remove_think_prefix": True,
        },
    }
    ollama_embedder = {
        "backend": "ollama",
        "config": {
            "model_name_or_path": EMBED_MODEL,
            "api_base": OLLAMA_BASE,
        },
    }

    mos_config = MOSConfig(
        user_id=USER_ID,
        chat_model=ollama_llm,
        mem_reader={
            "backend": "simple_struct",
            "config": {
                "llm": ollama_llm,
                "embedder": ollama_embedder,
                "chunker": {
                    "backend": "sentence",
                    "config": {
                        "tokenizer_or_token_counter": "gpt2",
                        "chunk_size": 512,
                        "chunk_overlap": 128,
                        "min_sentences_per_chunk": 1,
                    },
                },
            },
        },
        enable_textual_memory=True,
        enable_activation_memory=False,
        top_k=5,
        max_turns_window=20,
        enable_mem_scheduler=False,
    )

    cube_config = GeneralMemCubeConfig(
        user_id=USER_ID,
        cube_id=f"{USER_ID}_cube",
        text_mem={
            "backend": "tree_text",
            "config": {
                "extractor_llm": ollama_llm,
                "dispatcher_llm": ollama_llm,
                "graph_db": {
                    "backend": "neo4j",
                    "config": {
                        "uri": NEO4J_URI,
                        "user": NEO4J_USER,
                        "password": NEO4J_PASS,
                        "db_name": NEO4J_DB,
                        "auto_create": False,
                        "use_multi_db": False,
                        "embedding_dimension": EMBED_DIM,
                        "user_name": f"memos{USER_ID.replace('-', '').replace('_', '')}",
                    },
                },
                "embedder": ollama_embedder,
                "reorganize": False,
            },
        },
        act_mem={},
        para_mem={},
    )

    cube = GeneralMemCube(cube_config)
    mos = MOS(config=mos_config)
    mos.register_mem_cube(cube)
    return mos


# ── Collect files ─────────────────────────────────────────────────────────────
def collect_files(root):
    files = []
    for dirpath, _, filenames in os.walk(root):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXT:
                files.append(os.path.join(dirpath, fname))
    return sorted(files)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Check dependencies upfront
    missing = []
    try:
        import pypdf
    except ImportError:
        missing.append("pypdf")
    try:
        import docx
    except ImportError:
        missing.append("python-docx")

    if missing:
        print(f"Missing dependencies. Run: pip install {' '.join(missing)}")
        sys.exit(1)

    print("Building MOS connection to MemCore...")
    mos = build_mos()
    print("Connected.\n")

    files = collect_files(CORPUS_ROOT)
    total = len(files)
    print(f"Found {total} files to ingest.\n")

    ok, failed, skipped = 0, [], []

    for i, filepath in enumerate(files, 1):
        rel = os.path.relpath(filepath, CORPUS_ROOT)
        print(f"[{i}/{total}] {rel} ... ", end="", flush=True)
        try:
            text = extract_text(filepath)
            if not text:
                print("SKIPPED (empty)")
                skipped.append(rel)
                continue
            # Prefix with filename so the memory knows its source
            content = f"[Source: {rel}]\n\n{text}"
            mos.add(memory_content=content, user_id=USER_ID)
            print("OK")
            ok += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed.append((rel, str(e)))

    print(f"\n── Done ──────────────────────────────")
    print(f"  Ingested: {ok}/{total}")
    if skipped:
        print(f"  Skipped:  {len(skipped)} (empty content)")
    if failed:
        print(f"  Failed:   {len(failed)}")
        for name, err in failed:
            print(f"    • {name}: {err}")


if __name__ == "__main__":
    main()
