#!/usr/bin/env python3
"""
Wrapper script to run visualize_db_argilla from the src directory.
This allows running the script from the project root while maintaining the src structure.
"""

if __name__ == "__main__":
    from src.visualize_db_argilla import main
    main()