import json
import random

# Generate synthetic training data
portfolio = {
    "assets": [
        {"name": "Infra", "weight": 0.2, "returns": 0.044},
        {"name": "Home Loans", "weight": 0.1, "returns": 0.05},
        {"name": "Public Equity", "weight": 0.3, "returns": 0.08},
        {"name": "Private Equity", "weight": 0.05, "returns": 0.07},
        {"name": "Commodity", "weight": 0.15, "returns": 0.04},
        {"name": "Liquid Cash", "weight": 0.2, "returns": 0.0}
    ],
    "returns": []
}

for _ in range(5):
    annual_return = {}
    for asset in portfolio["assets"]:
        asset_return = asset["returns"] + random.uniform(-0.05, 0.05)
        annual_return[asset["name"]] = asset_return
    portfolio["returns"].append(annual_return)

# Save the synthetic training data to a JSON file
with open("training_data.json", "w") as file:
    json.dump(portfolio, file, indent=4)

print("Synthetic training data generated and saved to 'training_data.json' file.")
