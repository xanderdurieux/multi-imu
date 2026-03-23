"""Compatibility wrapper exposing pipeline comparison helpers."""

from .pipeline import apply_best_method, compare_methods, main, select_best_method

__all__ = ['apply_best_method', 'compare_methods', 'select_best_method', 'main']


if __name__ == '__main__':
    main()
