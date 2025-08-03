#!/usr/bin/env python3
"""
Wrapper script to run populate_database from the src directory.
This allows running the script from the project root while maintaining the src structure.
"""

if __name__ == "__main__":
    from src.populate_database import main
    main()