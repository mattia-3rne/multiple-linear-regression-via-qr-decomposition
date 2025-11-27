import pandas as pd
import numpy as np

# Setup data generation
n_samples = 100

# Generating features
square_meters = np.random.randint(40, 300, size=n_samples)
num_rooms = np.maximum(2, np.ceil(square_meters / 60) + np.random.randint(-1, 2, size=n_samples)).astype(int)
loc_score = np.random.randint(1, 6, size=n_samples)
age_years = np.random.randint(0, 50, size=n_samples)

# Creating target price using a linear formula + some noise
base_price = 50000
noise = np.random.normal(0, 15000, size=n_samples)
price = base_price + (2500 * square_meters) + (15000 * num_rooms) + (25000 * loc_score) + (-750 * age_years) + noise

# Saving to CSV
df = pd.DataFrame({
    "square_meters": square_meters,
    "num_rooms": num_rooms,
    "loc_score": loc_score,
    "age_years": age_years,
    "price": price.astype(int)
})

df.to_csv("house_prices_dataset_linear.csv", index=False)
print("File 'house_prices_dataset_linear.csv' created successfully!")
