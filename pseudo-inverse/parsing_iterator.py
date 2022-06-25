import subprocess, os, sys

lst = ["testcsv.csv", "VIE_bashirinslsl_och.csv"]

for file in lst:
    print(f"\n{file}")
    sub_process = subprocess.Popen(["python", "parser.py", file, file + "_adjusted.csv"])
    #wait
    sub_process.wait()