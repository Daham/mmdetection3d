import pickle
import sys

if len(sys.argv) != 2:
    print('Usage: python inspect_kitti_infos.py input.pkl')
    sys.exit(1)

infile = sys.argv[1]
with open(infile, 'rb') as f:
    infos = pickle.load(f)

print(f"Type of loaded object: {type(infos)}")
if isinstance(infos, dict):
    print(f"Keys: {list(infos.keys())}")
    info_list = infos.get('infos', [])
else:
    info_list = infos

print(f"Type of first 5 entries in info_list:")
for i, entry in enumerate(info_list[:5]):
    print(f"  Entry {i}: {type(entry)}")
    if isinstance(entry, dict):
        print(f"    Keys: {list(entry.keys())}")
    else:
        print(f"    Value: {entry}")