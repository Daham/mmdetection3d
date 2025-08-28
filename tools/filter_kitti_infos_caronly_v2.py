import pickle
import sys

if len(sys.argv) != 3:
    print('Usage: python filter_kitti_infos_caronly_v2.py input.pkl output.pkl')
    sys.exit(1)

infile, outfile = sys.argv[1:3]
with open(infile, 'rb') as f:
    infos = pickle.load(f)

if not (isinstance(infos, dict) and 'data_list' in infos):
    print('ERROR: Unexpected file format. Expected dict with key "data_list".')
    sys.exit(1)

filtered_list = []
for entry in infos['data_list']:
    if not isinstance(entry, dict):
        continue
    annos = entry.get('annos', {})
    names = annos.get('name', [])
    # If there are no objects, skip
    if not names:
        continue
    # Mask for 'Car' objects
    mask = [n == 'Car' for n in names]
    if not any(mask):
        continue
    # Filter all annotation fields
    for k, v in annos.items():
        if isinstance(v, list):
            annos[k] = [x for x, m in zip(v, mask) if m]
        elif hasattr(v, '__len__') and not isinstance(v, str):
            annos[k] = [x for x, m in zip(v, mask) if m]
    entry['annos'] = annos
    filtered_list.append(entry)

out_dict = dict(metainfo=infos.get('metainfo', {}), data_list=filtered_list)
with open(outfile, 'wb') as f:
    pickle.dump(out_dict, f)

print(f'Filtered {infile} -> {outfile} (Car only, new format)')
