#!/bin/bash
echo "Installing required Python packages..."
pip3 install -r requirements.txt
echo "Starting IntelliDoc AI Application..."
streamlit run app.py
