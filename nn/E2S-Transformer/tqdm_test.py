from tqdm import tqdm, trange
import time

for i1 in tqdm(range(100)):
    time.sleep(0.01)
    if (i1+1) % 50 == 0:
        with tqdm(total=100) as pbar:
            for i2 in range(150):
                # do something, e.g. sleep
                time.sleep(0.01)
                pbar.update(1)