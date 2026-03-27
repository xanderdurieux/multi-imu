"""Legacy convenience entrypoint.

Prefer ``python -m workflow`` (config-driven) or ``python -m pipeline``.
"""


def main() -> None:
    from workflow.run import main as workflow_main

    workflow_main()


if __name__ == "__main__":
    main()
