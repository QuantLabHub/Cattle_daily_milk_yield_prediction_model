#MODEL

# ============================================================
# JANU'S FARM - VETERINARY MACHINE LEARNING PROJECT
# 30-DAY CATTLE MILK YIELD PREDICTION SYSTEM
# Algorithm: Multiple Linear Regression (Log-Target)
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

# ============================================================
# 1. LOAD DATASET
# ============================================================

df = pd.read_csv("janus_farm_3_month_cattle_dataset.csv")

print("\n============================================")
print("       JANU'S FARM - CATTLE ML SYSTEM")
print("============================================")

print("Dataset loaded successfully!")
print("Total records:", len(df))
print("Total cattle:", df["ear_tag_id"].nunique())


# ============================================================
# 2. DATA PREPARATION
# ============================================================

required_columns = [
    "ear_tag_id",
    "breed",
    "breed_code",
    "age_years",
    "body_weight_kg",
    "daily_energy_intake_kcal",
    "daily_milk_yield_l"
]

df = df[required_columns].copy()

df = df.replace([np.inf, -np.inf], np.nan).dropna()

# Milk yield must be positive for logarithmic regression
df = df[df["daily_milk_yield_l"] > 0].copy()

df["breed_code"] = df["breed_code"].astype(int)

# Breed code mapping from CSV
breed_mapping = (
    df[["breed_code", "breed"]]
    .drop_duplicates()
    .sort_values("breed_code")
)

breed_dictionary = dict(
    zip(breed_mapping["breed_code"], breed_mapping["breed"])
)

# ============================================================
# 3. INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

X = df[
    [
        "breed",
        "age_years",
        "body_weight_kg",
        "daily_energy_intake_kcal"
    ]
]

y = df["daily_milk_yield_l"]

groups = df["ear_tag_id"]


# ============================================================
# 4. TRAIN-TEST SPLIT
# ============================================================

splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.20,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=groups)
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]


# ============================================================
# 5. BREED ENCODING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "breed_encoder",
            OneHotEncoder(handle_unknown="ignore"),
            ["breed"]
        )
    ],
    remainder="passthrough"
)


# ============================================================
# 6. CREATE LOGARITHMIC LINEAR REGRESSION MODEL
# ============================================================

# Log transformation guarantees positive predictions
# after applying the exponential function.

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regression", LinearRegression())
    ]
)

# Train using logarithm of milk yield
model.fit(X_train, np.log(y_train))


# ============================================================
# 7. MODEL EVALUATION
# ============================================================

log_predictions = model.predict(X_test)

# Convert predictions back to litres
y_pred = np.exp(log_predictions)

mae = mean_absolute_error(y_test, y_pred)

rmse = np.sqrt(
    mean_squared_error(y_test, y_pred)
)

r2 = r2_score(y_test, y_pred)

print("\n============================================")
print("            MODEL PERFORMANCE")
print("============================================")

print("MAE:", round(mae, 3), "litres")
print("RMSE:", round(rmse, 3), "litres")
print("R2 Score:", round(r2, 3))


# ============================================================
# 8. TRAIN FINAL MODEL USING ALL DATA
# ============================================================

model.fit(X, np.log(y))


# ============================================================
# 9. MENU-DRIVEN CATTLE REGISTRATION
# ============================================================

print("\n============================================")
print("          NEW CATTLE REGISTRATION")
print("============================================")

print("\nSELECT CATTLE BREED")

for code, breed_name in breed_dictionary.items():
    print(f"{code}. {breed_name}")


# Breed input

while True:
    try:
        breed_code = int(input("\nEnter Breed Code: "))

        if breed_code in breed_dictionary:
            selected_breed = breed_dictionary[breed_code]
            break
        else:
            print("Invalid code. Please select a displayed breed.")

    except ValueError:
        print("Please enter a valid numeric breed code.")


# Age input

while True:
    try:
        age = float(input("\nEnter cattle age (years): "))

        if np.isfinite(age) and age > 0:
            break
        else:
            print("Age must be a positive number.")

    except ValueError:
        print("Enter a valid age.")


# Body weight input

while True:
    try:
        weight = float(
            input("\nEnter body weight (kg): ")
        )

        if np.isfinite(weight) and weight > 0:
            break
        else:
            print("Body weight must be positive.")

    except ValueError:
        print("Enter a valid body weight.")


# Daily energy intake input

while True:
    try:
        energy = float(
            input("\nEnter daily energy intake (kcal): ")
        )

        if np.isfinite(energy) and energy > 0:
            break
        else:
            print("Energy intake must be positive.")

    except ValueError:
        print("Enter a valid energy intake.")


# ============================================================
# 10. DISPLAY CATTLE DETAILS
# ============================================================

print("\n============================================")
print("             CATTLE DETAILS")
print("============================================")

print("Breed:", selected_breed)
print("Breed Code:", breed_code)
print("Age:", age, "years")
print("Body Weight:", weight, "kg")
print("Daily Energy Intake:", energy, "kcal")


# Warn if values are outside the training data range

input_values = {
    "age_years": age,
    "body_weight_kg": weight,
    "daily_energy_intake_kcal": energy
}

print("\nDATA RANGE CHECK:")

for column, value in input_values.items():

    minimum = df[column].min()
    maximum = df[column].max()

    if value < minimum or value > maximum:
        print(
            f"WARNING: {column} is outside the "
            f"training range ({minimum:.2f} - {maximum:.2f})."
        )
        print("Prediction may be less reliable.")

    else:
        print(f"{column}: Within training range.")


# ============================================================
# 11. CREATE NEW CATTLE INPUT
# ============================================================

new_cow = pd.DataFrame([{

    "breed": selected_breed,
    "age_years": age,
    "body_weight_kg": weight,
    "daily_energy_intake_kcal": energy

}])


# ============================================================
# 12. PREDICT DAILY MILK YIELD
# ============================================================

log_yield = model.predict(new_cow)[0]

predicted_daily_yield = np.exp(log_yield)

print("\n============================================")
print("          MILK YIELD PREDICTION")
print("============================================")

print(
    "Predicted Daily Milk Yield:",
    round(predicted_daily_yield, 2),
    "litres/day"
)


# ============================================================
# 13. GENERATE 30-DAY MILK PREDICTION
# ============================================================

days = np.arange(1, 31)

# Constant input conditions:
# Same breed, age, weight and energy intake.
# Therefore, the model gives the same daily prediction.

daily_predictions = np.full(
    30,
    predicted_daily_yield
)

cumulative_predictions = np.cumsum(
    daily_predictions
)

forecast = pd.DataFrame({

    "Day": days,
    "Breed": selected_breed,
    "Age_Years": age,
    "Body_Weight_Kg": weight,
    "Daily_Energy_Intake_Kcal": energy,
    "Predicted_Milk_Yield_L": daily_predictions,
    "Cumulative_Milk_Yield_L": cumulative_predictions

})


# ============================================================
# 14. DISPLAY 30-DAY FORECAST
# ============================================================

print("\n============================================")
print("          30-DAY MILK FORECAST")
print("============================================")

print(
    forecast.to_string(
        index=False,
        formatters={
            "Predicted_Milk_Yield_L": "{:.2f}".format,
            "Cumulative_Milk_Yield_L": "{:.2f}".format
        }
    )
)


# ============================================================
# 15. PRODUCTION SUMMARY
# ============================================================

total_milk = cumulative_predictions[-1]

print("\n============================================")
print("             PRODUCTION SUMMARY")
print("============================================")

print("Daily Milk Yield:", round(predicted_daily_yield, 2), "L")

print("7-Day Milk Production:",
      round(daily_predictions[:7].sum(), 2), "L")

print("15-Day Milk Production:",
      round(daily_predictions[:15].sum(), 2), "L")

print("30-Day Milk Production:",
      round(total_milk, 2), "L")


# ============================================================
# 16. GRAPH 1 - DAILY MILK YIELD
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    days,
    daily_predictions,
    marker="o",
    linewidth=2,
    label="Predicted Daily Milk Yield"
)

plt.title(
    f"30-Day Daily Milk Yield Prediction - {selected_breed}"
)

plt.xlabel("Day")
plt.ylabel("Milk Yield (Litres)")

plt.xticks(np.arange(1, 31, 1))
plt.ylim(bottom=0)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 17. GRAPH 2 - CUMULATIVE MILK PRODUCTION
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    days,
    cumulative_predictions,
    marker="o",
    linewidth=2,
    label="Cumulative Milk Production"
)

plt.fill_between(
    days,
    cumulative_predictions,
    alpha=0.15
)

plt.title(
    f"30-Day Cumulative Milk Production - {selected_breed}"
)

plt.xlabel("Day")
plt.ylabel("Cumulative Milk (Litres)")

plt.xticks(np.arange(1, 31, 1))
plt.grid(alpha=0.3)

plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 18. GRAPH 3 - ACTUAL VS PREDICTED MODEL PERFORMANCE
# ============================================================

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    y_pred,
    alpha=0.6
)

lower = min(y_test.min(), y_pred.min())
upper = max(y_test.max(), y_pred.max())

plt.plot(
    [lower, upper],
    [lower, upper],
    "r--",
    label="Perfect Prediction"
)

plt.xlabel("Actual Milk Yield (Litres)")
plt.ylabel("Predicted Milk Yield (Litres)")

plt.title("Actual vs Predicted Milk Yield")

plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


# ============================================================
# 19. SAVE 30-DAY FORECAST
# ============================================================

forecast.to_csv(
    "JANUS_FARM_30_DAY_MILK_FORECAST.csv",
    index=False
)

print("\n============================================")
print("30-DAY FORECAST SAVED SUCCESSFULLY!")
print("JANUS_FARM_30_DAY_MILK_FORECAST.csv")
print("============================================")

print("\nThank you for using Janu's Farm ML System!")


# ============================================================4
# END OF PROJECT
# ============================================================

#OUTPUT IS BASED ON THE USER INPUTS OF CATTLE INFORNATION
