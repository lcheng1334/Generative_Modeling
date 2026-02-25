import os

base = r'E:\code\dataset\Generative_Modeling\data\datasets\NG\classify'
types = ['adhesion','breakage','contamination','diffusion',
         'exposed_substrate','reversed_print','silver_overflow','other']

print(f"{'Defect Type':20s} | {'Cam1':>4s} | {'Cam2':>4s} | {'Cam3':>4s} | {'Cam4':>4s} | {'Cam5':>4s} | {'Cam6':>4s} | {'Total':>5s}")
print("-" * 80)

grand_cams = {f'Cam{i}': 0 for i in range(1, 7)}
grand_total = 0

for t in types:
    d = os.path.join(base, t)
    if not os.path.isdir(d):
        continue
    files = os.listdir(d)
    cams = {}
    for i in range(1, 7):
        count = sum(1 for f in files if f.startswith(f'Cam{i}'))
        cams[f'Cam{i}'] = count
        grand_cams[f'Cam{i}'] += count
    total = len(files)
    grand_total += total
    print(f"{t:20s} | {cams['Cam1']:4d} | {cams['Cam2']:4d} | {cams['Cam3']:4d} | {cams['Cam4']:4d} | {cams['Cam5']:4d} | {cams['Cam6']:4d} | {total:5d}")

print("-" * 80)
print(f"{'TOTAL':20s} | {grand_cams['Cam1']:4d} | {grand_cams['Cam2']:4d} | {grand_cams['Cam3']:4d} | {grand_cams['Cam4']:4d} | {grand_cams['Cam5']:4d} | {grand_cams['Cam6']:4d} | {grand_total:5d}")
