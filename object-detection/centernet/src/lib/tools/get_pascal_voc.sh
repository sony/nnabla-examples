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

VOC_ROOT=voc/
ORIG_DIR=$(pwd)

mkdir $VOC_ROOT
cd $VOC_ROOT

# Download Pascal VOC image tar files.
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar

# Extract tar files and remove them.
tar xvf VOCtrainval_06-Nov-2007.tar
tar xvf VOCtest_06-Nov-2007.tar
tar xvf VOCdevkit_08-Jun-2007.tar
tar xvf VOCtrainval_11-May-2012.tar
tar xvf VOCdevkit_18-May-2011.tar
rm VOCtrainval_06-Nov-2007.tar
rm VOCtest_06-Nov-2007.tar
rm VOCdevkit_08-Jun-2007.tar
rm VOCtrainval_11-May-2012.tar
rm VOCdevkit_18-May-2011.tar

# Move images to $VOC_ROOT/images/
mkdir images
cp VOCdevkit/VOC2007/JPEGImages/* images/
cp VOCdevkit/VOC2012/JPEGImages/* images/

# Donwload PASCAL VOC annotations in COCO format and move them to $VOC_ROOT/annotations.
# Or, you also convert them yourself, for example in a similar way with https://github.com/soumenpramanik/Convert-Pascal-VOC-to-COCO/blob/master/convertVOC2COCO.py
wget https://s3.amazonaws.com/images.cocodataset.org/external/external_PASCAL_VOC.zip
unzip external_PASCAL_VOC.zip
rm external_PASCAL_VOC.zip
mv PASCAL_VOC annotations/


# Merge all Pascal VOC annotationos from train-val of 2007 & 2012.
cd $ORIG_DIR
python src/lib/tools/merge_pascal_json.py -a $VOC_ROOT/annotations
