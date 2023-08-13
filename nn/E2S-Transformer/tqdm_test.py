from tqdm.auto import tqdm, trange
import time

for i1 in tqdm(range(100)):
    time.sleep(0.01)
    if (i1+1) % 50 == 0:
        for i2 in tqdm(range(150)):
            # do something, e.g. sleep
            time.sleep(0.01)