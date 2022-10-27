import csv
import os

import matplotlib.pyplot as plt

x = []
y_itr = []
y_time = []

path = os.path.join(os.path.dirname(
    os.path.abspath(__file__)), 'data/step_size_h2o.csv')
with open(path, newline='') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        x.append(float(row[0]))
        y_itr.append(int(row[1]))
        y_time.append(float(row[2]))

plt.plot(x, y_itr, '.-')
plt.axhline(y=117, linestyle='--', color='orange')
plt.semilogx()
plt.xlabel("step size", fontsize=16)
plt.ylabel("number of iterations", fontsize=16)
plt.savefig("result-step-iteration.png", dpi=300)
plt.show()

plt.plot(x, y_time, '.-')
plt.axhline(y=719.1, linestyle='--', color='orange')
plt.semilogx()
plt.xlabel("step size", fontsize=16)
plt.ylabel("wall time (ms)", fontsize=16)
plt.savefig("result-step-time.png", dpi=300)
plt.show()
