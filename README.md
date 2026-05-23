# End-to-End Customer Churn MLOps
Run:
pip install -r requirements.txt
python src/train.py
mlflow ui
uvicorn src.api:app --reload
