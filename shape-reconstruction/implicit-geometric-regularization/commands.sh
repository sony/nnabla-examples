# Copyright 2020,2021 Sony Corporation.
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


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/dragon_recon/dragon_vrip_pcd.ply \
    -m dragon \
    -k 50 \
    --lam 0.01


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/drill/reconstruction/drill_shaft_vrip_pcd.ply \
    -m drill \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/happy_recon/
    -m drill \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/happy_recon/happy_vrip_pcd.ply \
    -m happy \
    -k 50 \
    --lam 0.1

python train.py -d 1 \
    -f stanford_3d_scanning_datasets/Armadillo_pcd.ply \
    -m armadillo \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/lucy_pcd.ply \
    -m lucy \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_dragon_pcd.ply \
    -m rgb_dragon \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_manuscript_pcd.ply \
    -m manuscript \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f stanford_3d_scanning_datasets/xyzrgb_statuette_pcd.ply \
    -m statuette \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/anchor.xyz \
    -m anchor \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/dc.xyz \
    -m dc \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/lord_quas.xyz \
    -m lord_quas \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/daratech.xyz \
    -m daratech \
    -k 50 \
    --lam 0.1


python train.py -d 1 \
    -f deep_geometric_prior_data/ground_truth/gargoyle.xyz \
    -m gargoyle \
    -k 50 \
    --lam 0.1
