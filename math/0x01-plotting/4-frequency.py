#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

plt.hist(student_grades, range=(0, 100), edgecolor='black')
plt.ylim(0, 30)
plt.xticks(np.arange(110, step=10))
plt.margins(0)
plt.xlabel('Grades')
plt.ylabel('Number of Students')
plt.title('Project A')
plt.show()
