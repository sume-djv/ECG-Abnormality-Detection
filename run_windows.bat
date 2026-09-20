@echo off
python -m pip install -r requirements.txt
python train.py
streamlit run app.py
