# Copyright 2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# --------------------------------------------------------
# Reference: https://github.com/xingyizhou/CenterNet
# --------------------------------------------------------

def get_args():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--annot-path', '-a', type=str, default='voc/annotations/',
                   help='A folder where annotation JSON files are located.')
    p.add_argument('--out-path', '-o', type=str, default=None,
                   help='A folder where a merged output JSON file will be generated. The default will be the same location as --annot-path.')
    args = p.parse_args()
    return args


def main():
    import json
    from os.path import join
    args = get_args()
    ANNOT_PATH = args.annot_path
    if args.out_path is None:
        args.out_path = args.annot_path
    OUT_PATH = args.out_path
    INPUT_FILES = ['pascal_train2012.json', 'pascal_val2012.json',
                   'pascal_train2007.json', 'pascal_val2007.json']
    OUTPUT_FILE = 'pascal_trainval0712.json'
    KEYS = ['images', 'type', 'annotations', 'categories']
    MERGE_KEYS = ['images', 'annotations']

    out = {}
    tot_anns = 0
    for i, file_name in enumerate(INPUT_FILES):
        data = json.load(open(join(ANNOT_PATH, file_name), 'r'))
        print('keys', data.keys())
        if i == 0:
            for key in KEYS:
                out[key] = data[key]
                print(file_name, key, len(data[key]))
        else:
            out['images'] += data['images']
            for j in range(len(data['annotations'])):
                data['annotations'][j]['id'] += tot_anns
            out['annotations'] += data['annotations']
            print(file_name, 'images', len(data['images']))
            print(file_name, 'annotations', len(data['annotations']))
        tot_anns = len(out['annotations'])
    print('tot', len(out['annotations']))
    json.dump(out, open(join(OUT_PATH, OUTPUT_FILE), 'w'))


if __name__ == '__main__':
    main()
