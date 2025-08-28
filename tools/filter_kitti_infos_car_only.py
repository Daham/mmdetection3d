import pickle
import sys
from pathlib import Path

def filter_infos(infile, outfile):
    with open(infile, 'rb') as f:
        infos = pickle.load(f)
    # Support both dict and list formats
    if isinstance(infos, dict) and 'infos' in infos:
        info_list = infos['infos']
        meta = {k: v for k, v in infos.items() if k != 'infos'}
    else:
        info_list = infos
        meta = None
    # Filter for 'Car' only
    filtered = []
    for info in info_list:
        if not isinstance(info, dict):
            continue  # skip non-dict entries
        annos = info.get('annos', {})
        if 'name' in annos:
            mask = [n == 'Car' for n in annos['name']]
            # If no car in this sample, skip
            if not any(mask):
                continue
            # Filter all annotation fields
            for k, v in annos.items():
                if isinstance(v, list):
                    annos[k] = [x for x, m in zip(v, mask) if m]
                elif hasattr(v, '__len__') and not isinstance(v, str):
                    annos[k] = [x for x, m in zip(v, mask) if m]
            info['annos'] = annos
        filtered.append(info)
    # Save in same format
    if meta is not None:
        out = dict(infos=filtered, **meta)
    else:
        out = filtered
    with open(outfile, 'wb') as f:
        pickle.dump(out, f)

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python filter_kitti_infos_car_only.py input.pkl output.pkl')
        sys.exit(1)
    infile, outfile = sys.argv[1:3]
    filter_infos(infile, outfile)
    print(f'Filtered {infile} -> {outfile} (Car only)')
