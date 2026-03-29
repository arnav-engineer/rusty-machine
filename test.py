import numpy as np
from rustymachine_api.models import LinearRegression
import cupy as cp

X = np.random.randn(10, 13)
y = np.random.randn(10)

model = LinearRegression(alpha=0.1)
model.fit(X, y)
print("Coef:", model.coef_)
print("Intercept:", model.intercept_)

preds = model.predict(X)
print("Preds:", preds)
