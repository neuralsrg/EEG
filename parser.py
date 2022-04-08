import numpy as np
from progress.bar import ChargingBar
import sys

csv_file = open(sys.argv[1], "r")
lines = csv_file.readlines()
csv_file.close()

res = np.empty(0)
row = np.empty(0)

bar = ChargingBar(f'Processing {sys.argv[1]}...', max=len(lines))

for i in range(len(lines)): 
    if (i%12 == 0):
        #i-th line. Then get rid of \n.
        array_line = np.fromstring(lines[i][:-1], sep=";")
        row = np.append(row, array_line.reshape(-1), axis = 0)
    else:
        #also get rid of ; at the beginning
        array_line = np.fromstring(lines[i][1:-1], sep=";")
        row = np.append(row, array_line.reshape(-1), axis = 0)

    #flush each 12th
    if (i % 12 == 11):
        res = np.append(res, row, axis = 0)
        row = np.empty(0)


    bar.next()

bar.finish()
res = res.reshape(-1, 1176)
print(f"Writing array of shape {res.shape} to {sys.argv[2]}...")
np.savetxt(sys.argv[2], res.reshape(-1, 1176), fmt = '%.6f', delimiter = ";")
print("Success!\n")