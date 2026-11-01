import os
import shutil

src_base = r'C:\Users\fuzail\Downloads\Quality-Aware-Face-Recognition-System-main-old\Quality-Aware-Face-Recognition-System-main\dataset'
dst_base = r'C:\Users\fuzail\Downloads\quality-aware-face-recognition-share\dataset'

identities = ['amir', 'salman', 'sharuk']

for ident in identities:
    src_dir = os.path.join(src_base, ident)
    dst_dir = os.path.join(dst_base, ident)
    os.makedirs(dst_dir, exist_ok=True)
    if not os.path.exists(src_dir):
        print(f'Error: {src_dir} does not exist!')
        continue
    for fname in os.listdir(src_dir):
        sf = os.path.join(src_dir, fname)
        df = os.path.join(dst_dir, fname)
        if os.path.isfile(sf) and not os.path.exists(df):
            shutil.copy2(sf, df)
            print(f'Copied: {ident}/{fname}')
        elif os.path.exists(df):
            print(f'Already exists: {ident}/{fname}')

print('Copy completed successfully.')
