from tqdm import tqdm, trange
import time
import random

for i1 in (pbar := tqdm(range(100))):
    pbar.set_description(f'T|loss:{random.random():.2f}|best val:{random.random():.2f}|cur val:{random.random():.2f}')
    pbar.set_postfix({'num_vowels': 78888888888881213})
    time.sleep(0.01)

print('a'*200)