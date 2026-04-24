import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules

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

encoder = TransactionEncoder()
encoded_array = encoder.fit(transactions).transform(transactions)

df = pd.DataFrame(encoded_array, columns=encoder.columns_)

frequent_itemsets = fpgrowth(
    df,
    min_support=0.3,
    use_colnames=True
)

print("Frequent Itemsets:")
print(frequent_itemsets)

rules = association_rules(
    frequent_itemsets,
    metric='confidence',
    min_threshold=0.6
)

rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]

print("\nAssociation Rules:")
print(rules)