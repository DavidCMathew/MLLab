import numpy as np
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination


# Read Cleveland Heart disease data
heartDisease = pd.read_csv('data/heart.csv')
heartDisease = heartDisease.replace('?', np.nan)

# Display the data
print('Few examples from the dataset are given below')
print(heartDisease.head())
print('\nAttributes and datatypes')
print(heartDisease.dtypes)

# The Bayesian network can be made taking any number of edges between any 2 nodes
model = BayesianModel([
                        ('age', 'trestbps'),            # trestbps = resting blood sugar
                        ('age', 'fbs'),                 # fbs = fasting blood sugar
                        ('sex', 'trestbps'),
                        ('trestbps', 'heartdisease'),
                        ('fbs', 'heartdisease'),
                        ('chol', 'heartdisease')        # chol = cholesterol
                        ])

# Learning CPDs using Maximum Likelihood Estimators
print('\nLearning CPDs using Maximum Likelihood Estimators...')
model.fit(heartDisease, estimator=MaximumLikelihoodEstimator)

# Deducing with Bayesian Network
print('\nInferencing with Bayesian Network:')
HeartDisease_infer = VariableElimination(model)

print('\n1.Probability of HeartDisease given Age=20')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 20})
print(q['heartdisease'])

print('\n2. Probability of HeartDisease given chol (Cholesterol) =100')
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'chol': 100})
print(q['heartdisease'])
