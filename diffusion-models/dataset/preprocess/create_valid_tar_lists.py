# Copyright 2022 Sony Group Corporation.
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

import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--blacklists", type=str, required=True, nargs="*")
    parser.add_argument("--datadirs", type=str, required=True, nargs="*")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    black_list = set()
    for path in args.blacklists:
        print(f"reading black list: {path}...")
        with open(path, "r") as f:
            lines = f.readlines()

        for line in lines:
            black_list.add(line.strip())

    files = []
    for path in args.datadirs:
        print(f"get all files in {path}...")

        for file in os.listdir(path):
            # skip if file is not tar
            if os.path.splitext(file)[-1] != ".tar":
                continue

            # skip if file in blacklist
            filepath = os.path.join(path, file)
            if filepath in black_list:
                continue

            files.append(filepath)

    with open(args.out, "w") as f:
        f.write("\n".join(files))


if __name__ == "__main__":
    main()
