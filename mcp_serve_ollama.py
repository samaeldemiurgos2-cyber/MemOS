#!/usr/bin/env python3
"""
MemOS MCP Server - Ollama + tree_text configuration for Claude Code.

This script starts the MemOS MCP server using:
  - Ollama as the LLM backend (native, no OpenAI key required)
  - Neo4j for tree_text memory (graph-structured memory)
  - Ollama embedder (nomic-embed-text:latest by default)

Prerequisites:
  1. Ollama running:  ollama serve
  2. Models pulled:   ollama pull llama3.2 && ollama pull nomic-embed-text
  3. Neo4j running:   see .env.claude.example for setup

Environment variables (set in ~/.claude/settings.json or a .env file):
  OLLAMA_API_BASE         Ollama server URL  (default: http://localhost:11434)
  OLLAMA_CHAT_MODEL       Chat model name    (default: llama3.2)
  OLLAMA_EMBEDDER_MODEL   Embedding model    (default: nomic-embed-text:latest)
  EMBEDDING_DIMENSION     Embedding dims     (default: 768 for nomic-embed-text)
  NEO4J_URI               Neo4j bolt URL     (default: bolt://localhost:7687)
  NEO4J_USER              Neo4j username     (default: neo4j)
  NEO4J_PASSWORD          Neo4j password     (required)
  NEO4J_DB_NAME           Neo4j database     (default: neo4j)
  MOS_USER_ID             Default user ID    (default: claude_user)
  MOS_CHAT_TEMPERATURE    LLM temperature    (default: 0.7)
  MOS_MAX_TOKENS          Max output tokens  (default: 2048)
"""

import argparse
import logging
import os
import sys
import warnings

# ── Redirect all output to stderr before importing MemOS ──────────────────────
# MCP stdio transport uses stdout exclusively for JSON-RPC messages.
# Any non-JSON text on stdout will break the protocol.
warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Force root logger to stderr at WARNING level before MemOS sets up its handlers
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

from dotenv import load_dotenv

# Load .env file if present (optional, env vars in settings.json take precedence)
load_dotenv()


def _redirect_handlers_to_stderr():
    """After all imports, ensure no logging handler writes to stdout."""
    for name in list(logging.Logger.manager.loggerDict.keys()):
        lgr = logging.getLogger(name)
        for handler in lgr.handlers:
            if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
                handler.stream = sys.stderr
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler) and getattr(handler, "stream", None) is sys.stdout:
            handler.stream = sys.stderr


def build_ollama_tree_config():
    """Build MOSConfig and GeneralMemCube configured for Ollama + tree_text."""
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.configs.mem_os import MOSConfig
    from memos.mem_cube.general import GeneralMemCube

    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    chat_model = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    embed_model = os.getenv("OLLAMA_EMBEDDER_MODEL", "nomic-embed-text:latest")
    user_id = os.getenv("MOS_USER_ID", "claude_user")

    neo4j_uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_db = os.getenv("NEO4J_DB_NAME", "neo4j")
    embed_dim = int(os.getenv("EMBEDDING_DIMENSION", "768"))  # nomic-embed-text = 768

    temperature = float(os.getenv("MOS_CHAT_TEMPERATURE", "0.7"))
    max_tokens = int(os.getenv("MOS_MAX_TOKENS", "2048"))

    ollama_llm = {
        "backend": "ollama",
        "config": {
            "model_name_or_path": chat_model,
            "api_base": ollama_base,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "remove_think_prefix": True,
        },
    }

    ollama_embedder = {
        "backend": "ollama",
        "config": {
            "model_name_or_path": embed_model,
            "api_base": ollama_base,
        },
    }

    mos_config = MOSConfig(
        user_id=user_id,
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
        user_id=user_id,
        cube_id=f"{user_id}_cube",
        text_mem={
            "backend": "tree_text",
            "config": {
                "extractor_llm": ollama_llm,
                "dispatcher_llm": ollama_llm,
                "graph_db": {
                    "backend": "neo4j",
                    "config": {
                        "uri": neo4j_uri,
                        "user": neo4j_user,
                        "password": neo4j_password,
                        "db_name": neo4j_db,
                        "auto_create": False,
                        "use_multi_db": False,
                        "embedding_dimension": embed_dim,
                        "user_name": f"memos{user_id.replace('-', '').replace('_', '')}",
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
    return mos_config, cube


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MemOS MCP Server (Ollama + tree_text)")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http", "sse"],
        default="stdio",
        help="Transport method (default: stdio)",
    )
    parser.add_argument("--host", default="localhost", help="Host for HTTP/SSE transport")
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP/SSE transport")
    args = parser.parse_args()

    from memos.api.mcp_serve import MOSMCPServer
    from memos.mem_os.main import MOS

    # Redirect any stdout log handlers to stderr after all imports
    _redirect_handlers_to_stderr()

    mos_config, cube = build_ollama_tree_config()
    mos = MOS(config=mos_config)
    mos.register_mem_cube(cube)

    server = MOSMCPServer(mos_instance=mos)
    server.run(transport=args.transport, host=args.host, port=args.port)
