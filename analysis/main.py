def main() -> None:
    """Delegate to the thesis pipeline CLI (``python -m pipeline`` preferred)."""
    from pipeline.run import main as pipeline_main

    pipeline_main()


if __name__ == "__main__":
    main()
