"""Legacy convenience entrypoint.

Prefer ``python -m workflow`` (config-driven).
"""


def main() -> None:
    import logging

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    logging.warning(
        "DEPRECATED: 'python main.py' is a legacy wrapper. "
        "Use 'python -m workflow --config configs/workflow.thesis.json'."
    )
    from workflow.run import main as workflow_main

    workflow_main()


if __name__ == "__main__":
    main()
