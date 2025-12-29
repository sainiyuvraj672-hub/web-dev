import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

df = pd.DataFrame({
    "Income": [25000, 48750, 20000, 50000, 100000]
})

# Min-Max Scaling
mm = MinMaxScaler()
df["Income_MinMax"] = mm.fit_transform(df[["Income"]])

# Z-Score Standardization
sc = StandardScaler()
df["Income_ZScore"] = sc.fit_transform(df[["Income"]])

print(df)
