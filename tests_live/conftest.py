"""conftest.py for tests_live/ — adds project root to sys.path."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
