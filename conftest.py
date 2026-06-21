"""Pytest configuration for the parking_intelligence package.

Ensures the project root is importable so `import parking_intelligence` works when
running `pytest` from anywhere, and registers a deterministic hypothesis profile so
property-based tests run reproducibly (supports the determinism requirements).
"""

import os
import sys

# Make the flat package importable regardless of the pytest invocation directory.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from hypothesis import settings

    # Deterministic, CI-friendly profile for property-based tests.
    settings.register_profile("ci", deadline=None, derandomize=True)
    settings.load_profile("ci")
except ImportError:  # hypothesis not installed yet; tests that need it will skip/fail clearly.
    pass
