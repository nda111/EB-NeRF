import os


with os.popen('nvidia-smi') as proc:
    lines = proc.readlines()
    lines.reverse()
    
    count = 0
    for i, line in enumerate(lines):
        if line[0] == '+':
            count += 1
            if count == 2:
                from_here = lines[:i + 1]
                from_here.reverse()
                print(''.join(from_here))
                exit()
