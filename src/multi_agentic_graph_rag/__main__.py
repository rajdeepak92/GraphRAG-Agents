"""Support execution through python -m multi_agentic_graph_rag."""

from .cli import app


def main() -> None:
    """Run the MARAG command-line application."""

    app()


if __name__ == "__main__":
    main()
