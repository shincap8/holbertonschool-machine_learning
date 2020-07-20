#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

columns = ['Farrah', 'Fred', 'Felicia']
rows = ['apples', 'bananas', 'oranges', 'peaches']
colors = ['r', '#ffff00', '#ff8000', '#ffe5b4']
y_offset = np.zeros(len(columns))

for i in range(len(fruit)):
    plt.bar(columns, fruit[i], width=0.5, color=colors[i], label=rows[i], bottom=y_offset)
    y_offset += fruit[i]

plt.ylim(0, 80)
plt.xticks(range(len(columns)), columns)
plt.legend(loc='upper right')
plt.ylabel('Quantity of Fruit')
plt.title('Number of Fruit per Person')
plt.show()
