"""Compatibility wrapper exposing the new pipeline entry points."""

from .pipeline import main, run_pipeline, run_session

__all__ = ['run_pipeline', 'run_session', 'main']


if __name__ == '__main__':
    main()
