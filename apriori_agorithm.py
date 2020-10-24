import os
import glob
import pandas as pd
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import matplotlib.pyplot as plt

df = pd.read_csv('store_products.csv')

def encode_units(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1
    
basket = df.groupby(['txn','category']).size().reset_index(name='count')
basket = (basket.groupby(['txn', 'category'])['count']
          .sum().unstack().reset_index().fillna(0)
          .set_index('txn'))
basket_sets = basket.applymap(encode_units)

frequent_itemsets = apriori(basket_sets, min_support=0.05, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
rules.sort_values('confidence', ascending = False, inplace = True)

rules = rules.reset_index(drop=True)

rules.to_csv("results.csv", index = False)

plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()