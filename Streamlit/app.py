'''
Imports
'''

import sys
import os
import streamlit as st

# Add the project's root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from predict import show_predict_page


# Execute streamlit from predict.py
if __name__ == "__main__":
    show_predict_page()
