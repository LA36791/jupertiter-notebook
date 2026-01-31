final_df.groupby("cuisine")["total_amount"].mean().sort_values(ascending=False)
print("Average total amount spent per cuisine:")
print(final_df.groupby("cuisine")["total_amount"].mean().sort_values(ascending=False).round(2))