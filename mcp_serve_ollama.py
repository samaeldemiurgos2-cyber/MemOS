#!/usr/bin/env python3
"""
MemOS MCP Server - Ollama + tree_text configuration.

Intended to run as a persistent SSE service on MemCore so that Claude Desktop,
Claude Code, and the phone app all connect via a single URL with no process
spawning and no stdio conflicts.

Deployment (MemCore):
    python mcp_serve_ollama.py --transport sse --host 0.0.0.0 --port 8000

Claude Desktop / Claude Code connection (Malfurion, phone):
    "url": "http://100.72.225.62:8000/sse"

Environment variables (set in .env or the systemd unit):
  OLLAMA_API_BASE         Ollama server URL  (default: http://localhost:11434)
  OLLAMA_CHAT_MODEL       Chat model name    (default: llama3.2)
  OLLAMA_EMBEDDER_MODEL   Embedding model    (default: nomic-embed-text:latest)
  EMBEDDING_DIMENSION     Embedding dims     (default: 768)
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

warnings.filterwarnings("ignore")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.basicConfig(stream=sys.stderr, level=logging.WARNING, force=True)

from dotenv import load_dotenv

load_dotenv()


def build_ollama_tree_config():
    from memos.configs.mem_cube import GeneralMemCubeConfig
    from memos.configs.mem_os import MOSConfig
    from memos.mem_cube.general import GeneralMemCube

    ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    chat_model  = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")
    embed_model = os.getenv("OLLAMA_EMBEDDER_MODEL", "nomic-embed-text:latest")
    user_id     = os.getenv("MOS_USER_ID", "claude_user")

    neo4j_uri      = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user     = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")
    neo4j_db       = os.getenv("NEO4J_DB_NAME", "neo4j")
    embed_dim      = int(os.getenv("EMBEDDING_DIMENSION", "768"))

    temperature = float(os.getenv("MOS_CHAT_TEMPERATURE", "0.7"))
    max_tokens  = int(os.getenv("MOS_MAX_TOKENS", "2048"))

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
        default="sse",
        help="Transport method (default: sse)",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    args = parser.parse_args()

    from memos.api.mcp_serve import MOSMCPServer
    from memos.mem_os.main import MOS

    mos_config, cube = build_ollama_tree_config()
    mos = MOS(config=mos_config)
    mos.register_mem_cube(cube)

    print(f"MemOS MCP server starting on {args.host}:{args.port} [{args.transport}]", file=sys.stderr)
    server = MOSMCPServer(mos_instance=mos)
    server.run(transport=args.transport, host=args.host, port=args.port)
