import pandas as pd

try:
    final_df = pd.read_csv('final_food_delivery_dataset.csv')
except FileNotFoundError:
    raise FileNotFoundError("final_food_delivery_dataset.csv not found. Run merge.py first to generate it.")

# ensure numeric and handle missing values
final_df['total_amount'] = pd.to_numeric(final_df.get('total_amount', 0), errors='coerce').fillna(0)

gold_city_revenue = (
    final_df[final_df["membership"] == "Gold"]
    .groupby("city")["total_amount"]
    .sum()
    .sort_values(ascending=False)
).round(2)

print("Total amount spent by Gold members per city:")
print(gold_city_revenue)

final_df.groupby("cuisine")["total_amount"].mean().sort_values(ascending=False)
