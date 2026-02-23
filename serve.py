"""Entry point for the AntiTerror API server.

Usage:
    python serve.py
    python serve.py --host 0.0.0.0 --port 8000
"""
import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="AntiTerror API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8000, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    uvicorn.run(
        "anti_terror.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
