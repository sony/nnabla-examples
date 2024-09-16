import argparse
import os
import subprocess
import sys
from pathlib import Path


def extract_video_frames(input_path):
    path = Path(input_path)

    frames_dir = os.path.join(path.parent, path.stem + "_frames")
    first_image_path = os.path.join(frames_dir, path.stem + "_000001.jpg")
    if os.path.isfile(first_image_path):
        print(
            "\nSkipping frame extraction for {} because frames seem"
            " to be extracted for this video already".format(path.stem)
        )
    else:
        # Make a folder for the frames, if the folder does not already exist
        os.makedirs(frames_dir, exist_ok=True)

        subprocess.run(
            [
                "ffmpeg",
                "-i",
                "{}".format(path.as_posix()),
                "{}".format(
                    Path(os.path.join(frames_dir, path.stem + "_%06d.jpg")).as_posix()
                ),
            ]
        )


def parse_args(args):
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--input-path",
        dest="input_path",
        help="Path to an MP4 video file",
        type=str,
        required=True,
    )
    return arg_parser.parse_args(args)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    extract_video_frames(args.input_path)