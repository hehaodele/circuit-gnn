import os

fn_list = os.listdir('./data/BPF200_400_3OSLR_Template_V2_ICML_Jan_22_1')

num = 3

cnt = {str(k):0 for k in range(10)}

for fn in fn_list:
    tp = fn[0]
    new_folder = f'./data/num{num}_type{tp}'
    os.system(f'mkdir -p {new_folder}')

    cnt[tp] += 1

    new_fn = f'{cnt[tp]:05d}.pkl'

    cmd = f'cp ./data/BPF200_400_3OSLR_Template_V2_ICML_Jan_22_1/{fn} {new_folder}/{new_fn}'

    os.system(cmd)