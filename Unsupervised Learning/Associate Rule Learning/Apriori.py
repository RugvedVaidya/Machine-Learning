import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Create transaction dataset
transactions = [
    ['Milk', 'Bread', 'Butter'],
    ['Bread', 'Butter'],
    ['Milk', 'Bread'],
    ['Milk', 'Butter'],
    ['Bread', 'Butter', 'Jam'],
    ['Milk', 'Bread', 'Butter', 'Jam'],
    ['Milk', 'Jam'],
    ['Bread', 'Jam']
]

# Step 2: Convert transactions into one-hot encoded format
encoder = TransactionEncoder()

encoded_array = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(
    encoded_array,
    columns=encoder.columns_
)

# Step 3: Print one-hot encoded dataset
print("Transaction Dataset:")
print(df)

# Step 4: Apply Apriori algorithm
frequent_itemsets = apriori(
    df,
    min_support=0.3,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_itemsets)

# Step 5: Generate association rules
rules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.6
)

# Step 6: Select useful columns
rules = rules[[
    'antecedents',
    'consequents',
    'support',
    'confidence',
    'lift'
]]

print("\nAssociation Rules:")
print(rules)