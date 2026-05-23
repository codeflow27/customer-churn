
from prefect import flow
from train import *
@flow
def churn_pipeline():
    print('running pipeline')
