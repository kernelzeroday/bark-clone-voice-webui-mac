import argparse
from typing import Dict, Optional, Union
import os

from scipy.io.wavfile import write as write_wav
from .api import generate_audio
from .generation import SAMPLE_RATE


def cli():
    """Commandline interface."""
    parser = argparse.ArgumentParser(description='Bark Audio Generation CLI', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--text", type=str, help="Text to be turned into audio.")
    parser.add_argument(
        "--output_filename",
        type=str,
        default="bark_generation.wav",
        help="Name of the output audio file.",
    )
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save the outputs.")
    parser.add_argument(
        "--history_prompt",
        type=str,
        default=None,
        help="History choice for audio cloning, should be path to the .npz file.",
    )
    parser.add_argument(
        "--text_temp",
        default=0.7,
        type=float,
        help="Generation temperature for text (1.0 more diverse, 0.0 more conservative).",
    )
    parser.add_argument(
        "--waveform_temp",
        default=0.7,
        type=float,
        help="Generation temperature for waveform (1.0 more diverse, 0.0 more conservative).",
    )
    parser.add_argument("--silent", default=False, type=bool, help="Disable progress bar if set to True.")
    parser.add_argument(
        "--output_full",
        default=False,
        type=bool,
        help="Return full generation to be used as a history prompt if set to True.",
    )

    args = vars(parser.parse_args())
    input_text: str = args.get("text")
    output_filename: str = args.get("output_filename")
    output_dir: str = args.get("output_dir")
    history_prompt: str = args.get("history_prompt")
    text_temp: float = args.get("text_temp")
    waveform_temp: float = args.get("waveform_temp")
    silent: bool = args.get("silent")
    output_full: bool = args.get("output_full")

    try:
        os.makedirs(output_dir, exist_ok=True)
        generated_audio = generate_audio(
            input_text,
            history_prompt=history_prompt,
            text_temp=text_temp,
            waveform_temp=waveform_temp,
            silent=silent,
            output_full=output_full,
        )
        output_file_path = os.path.join(output_dir, output_filename)
        write_wav(output_file_path, SAMPLE_RATE, generated_audio)
        print(f"Done! Output audio file is saved at: '{output_file_path}'")
    except Exception as e:
        print(f"Oops, an error occurred: {e}")

