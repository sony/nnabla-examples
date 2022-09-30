# Preprocess scripts for dataset preparation

## requirements
Run below before using scripts in this directory.

```
pip install -r requirements.txt
```

## LAION 400m

LAION-400m is an open dataset containing 400 million English image-text paires.
You can download the data from [the official project page](https://laion.ai/blog/laion-400-open-dataset/).

After downloading dataset, you should do below in order to use it with nnabla-examples/diffusion-models:
* make a list of path for all tarfiles
* encode texts with T5-XXL model 

Note that we rely on Hugging Face to use T5-XXL model. See [the official doc](https://huggingface.co/docs/transformers/model_doc/t5) for more details.

To make a list of path for all tarfiles, run below.
```
python create_valid_tar_lists.py \
    --blacklists {path/to/blacklist.txt} \
    --datadirs {path/to/datadir} \
    --out {/path/to/output.txt}
```
Each argument stands for:
* `--blacklists`: A list of text file to specify tarfiles to be ignored.
* `--datadirs`: A list of path for the data directories to be loaded. If you split data in several directories, you should specify all of them.
* `--out`: An output text file which will contain all tarfiles to be loaded.

Then, you should augment all tarfiles by computing T5 embedding for texts. 
Run below to do this.
```
python augment_laion_wds_t5_embed.py \
    --overwrite \
    --tarlists {/path/to/list_of_tarfiles.txt} \
    --outdir {/path/to/output_dir} \
    --logfile {/path/to/logfile.txt}
```
Each argument stands for:
* `--overwrite`: Input tarfiles will be replaced by augmented ones if specified.
* `--tarlists`: A list of path for all tarfiles. You can specify the output text file of `create_valid_tar_lists.py`.
* `--outdir`: A path for output directory. If overwrite is specified, each augmented tarfile will be place in this `outdir` once and moved to the original path.
* `--logfile`: A path for logging.