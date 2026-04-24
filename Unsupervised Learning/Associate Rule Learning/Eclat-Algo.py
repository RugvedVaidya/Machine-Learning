import pandas as pd
from itertools import combinations

# Dataset
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

# Convert to vertical format (TID sets)
tid_sets = {}

for tid, transaction in enumerate(transactions):
    for item in transaction:
        if item not in tid_sets:
            tid_sets[item] = set()
        tid_sets[item].add(tid)

# Minimum support
min_support = 0.3
min_count = int(min_support * len(transactions))

# Find frequent 1-itemsets
frequent_itemsets = {}

for item, tids in tid_sets.items():
    if len(tids) >= min_count:
        frequent_itemsets[frozenset([item])] = tids

# Generate larger itemsets
items = list(tid_sets.keys())

for i in range(2, 4):  # up to size 3
    for combo in combinations(items, i):
        intersect = set.intersection(*(tid_sets[item] for item in combo))
        if len(intersect) >= min_count:
            frequent_itemsets[frozenset(combo)] = intersect

# Print results
print("Frequent Itemsets:")
for itemset, tids in frequent_itemsets.items():
    support = len(tids) / len(transactions)
    print(set(itemset), "-> Support:", support)