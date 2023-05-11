
import os
import webdataset as wds
import braceexpand
import argparse
from threading import Thread
from queue import Queue
from tqdm import tqdm

from mpi4py import MPI

from transformers import T5Tokenizer, T5EncoderModel
import torch

# avoid error by PIL when loading bigger image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

T5_MODEL = "t5-11b"
t5_tokenizer = T5Tokenizer.from_pretrained(T5_MODEL)
t5_model = T5EncoderModel.from_pretrained(T5_MODEL)


def t5_encode(sentence, device):
    tokens = t5_tokenizer(sentence,
                          padding=True,
                          truncation=True,
                          max_length=1024,
                          return_length=True,
                          return_tensors="pt")

    outputs = t5_model(input_ids=tokens.input_ids.to(f"cuda:{device}"))

    emb_pt = outputs.last_hidden_state.detach()
    length = tokens.length.detach()

    return emb_pt, length


def save_data(q: Queue, file_path):
    with wds.TarWriter(file_path) as dst:
        while True:
            key, img, json, emb = q.get()
            if key == "STOP":
                break

            sample = {
                "__key__": key,
                "jpg": img,
                "json": json,
                "npz": {"t5_emb": emb.cpu().numpy()}
            }
            dst.write(sample)


def augment_wds(input, output_dir, batch_size, device, overwrite, logfile):
    """
    Given a single tarfile, 
    output a tarfile augemented by t5 embedding vector of numpy for a caption.

    Args:
        input (str): tarfile to be read.
        output_dir (str): A path for output directory.
        batch_size (int): A batchsize for t5 encode. 
        device (int): GPU device to run.
        overwrite (boolean): If True, the input tar file will be overwritten by an augmented tar file.
    """
    os.makedirs(output_dir, exist_ok=True)

    src = wds.DataPipeline(
        wds.SimpleShardList(input),
        wds.tarfile_to_samples(),
        wds.decode("rgb"),
        wds.to_tuple("__key__", "jpg", "json")
    )

    filename = os.path.basename("_".join(input.split("/")[-2:]))
    filepath = os.path.join(output_dir, filename)

    # setup thread to save data to tar
    q = Queue()
    thread = Thread(target=save_data, args=(q, filepath))

    # handle keyboard interrupt
    try:
        thread.start()

        # for saving
        batch_caption = []
        batch_others = []
        for key, img, json in tqdm(src, disable=device > 0):
            # get caption
            caption = json["caption"]
            if not isinstance(caption, str):
                continue

            # make data as a batch
            batch_others.append([key, img, json])
            batch_caption.append(caption)

            # if batch size is not enough, load next data
            if len(batch_caption) < batch_size:
                continue

            # encode captions
            with torch.inference_mode():
                emb, length = t5_encode(batch_caption, device)

            for i in range(batch_size):
                batch_others[i].append(emb[i, :length[i]])
                q.put(batch_others[i])

            batch_caption.clear()
            batch_others.clear()

        if len(batch_caption) > 0:
            with torch.inference_mode():
                emb, length = t5_encode(batch_caption, device)

            for i in range(len(batch_caption)):
                batch_others[i].append(emb[i, :length[i]])
                q.put(batch_others[i])

    except KeyboardInterrupt:
        # if interrupted by keyboard, just exit.
        q.put(("STOP", None, None, None))
        thread.join()
        exit(0)

    except Exception as e:
        # if any error happens, stop thread and show error message before exit.
        print(e)
        q.put(("STOP", None, None, None))
        thread.join()

        # dump error log and skip overwrite
        logfile.write(f"{input} \n")
        return

    # stop thread
    q.put(("STOP", None, None, None))
    thread.join()

    if overwrite:
        import shutil
        print(f"[overwrite] move {filepath} -> {input}")
        shutil.move(filepath, input)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--datadir", type=str)
    parser.add_argument("--start_id", default=0)
    parser.add_argument("--end_id", default=41407)
    parser.add_argument("--tarlists", default=None, type=str, nargs="*")
    parser.add_argument("--logfile", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    args = parser.parse_args()

    device = MPI.COMM_WORLD.Get_rank()
    t5_model.half().to(f"cuda:{device}")

    if args.tarlists is not None:
        tarfiles = []
        for tarlist in args.tarlists:
            assert os.path.exists(tarlist)

            with open(tarlist, "r") as f:
                lines = f.readlines()

            for line in lines:
                tarfile = line.strip()
                assert os.path.exists(tarfile)
                tarfiles.append(tarfile)

    else:
        args.datadir = os.path.abspath(args.datadir)
        assert os.path.exists(args.datadir)

        tarfiles = os.path.join(
            args.datadir, "{" + f"{int(args.start_id):05}..{int(args.end_id):05}" + "}.tar")
        tarfiles = list(braceexpand.braceexpand(tarfiles))

    from neu.datasets import get_slice_start_end
    start, end = get_slice_start_end(
        len(tarfiles), MPI.COMM_WORLD.Get_size(), device)

    with open(args.logfile, "w") as f:
        for tarfile in tqdm(tarfiles[start:end], disable=device > 0):
            augment_wds(tarfile,
                        args.outdir,
                        args.batch_size,
                        device,
                        args.overwrite,
                        f)
