import os
import sys

# Get the directory containing this file
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the parent directory (src)
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to Python path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
