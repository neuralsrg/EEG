from tqdm import tqdm, trange
import time
import random

for i1 in (pbar := tqdm(range(100), bar_format="{desc:<45}{percentage:3.0f}%|{bar:10}{r_bar:20}")):
    pbar.set_description(f'T|loss:{random.random():.2f}|best val:{random.random():.2f}|cur val:{random.random():.2f}')
    time.sleep(0.01)