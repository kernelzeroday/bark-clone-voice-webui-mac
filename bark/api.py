from typing import Dict, Optional, Union

import numpy as np

from .generation import codec_decode, generate_coarse, generate_fine, generate_text_semantic
from ..cloning.clonevoice import clone_voice
from ..swap_voice import swap_voice_from_audio


def create_clone_voice(audio_filepath, tokenizer_lang, dest_filename, progress=None):
    """Clone a voice based on provided voice data.

    Args:
        audio_filepath: Path to the audio file to be cloned.
        tokenizer_lang: Language of the tokenizer.
        dest_filename: Destination filename for the cloned voice.
        progress: Progress tracker.

    Returns:
        Cloned voice.
    """
    return clone_voice(audio_filepath, tokenizer_lang, dest_filename, progress)

def generate_swapped_voice(swap_audio_filename, selected_speaker, tokenizer_lang, seed, batchcount, progress=None):
    """Generate a swapped voice based on provided voice data and selected speaker.

    Args:
        swap_audio_filename: Path to the audio file to be swapped.
        selected_speaker: The selected speaker for voice swapping.
        tokenizer_lang: Language of the tokenizer.
        seed: Seed for random number generator.
        batchcount: Batch count for voice generation.
        progress: Progress tracker.

    Returns:
        Swapped voice.
    """
    return swap_voice_from_audio(swap_audio_filename, selected_speaker, tokenizer_lang, seed, batchcount, progress)


def generate_with_settings(text_prompt, semantic_temp=0.6, eos_p=0.2, coarse_temp=0.7, fine_temp=0.5, voice_name=None, output_full=False):
    """
    Generate text using the specified settings.

    Args:
        text_prompt (str): The text prompt to generate from.
        semantic_temp (float, optional): The temperature for semantic generation. Defaults to 0.6.
        eos_p (float, optional): The minimum end-of-sentence probability for semantic generation. Defaults to 0.2.
        coarse_temp (float, optional): The temperature for coarse generation. Defaults to 0.7.
        fine_temp (float, optional): The temperature for fine generation. Defaults to 0.5.
        voice_name (str, optional): The voice name to use for generation. Defaults to None.
        output_full (bool, optional): Whether to output the full generation or just the decoded fine prompt. Defaults to False.

    Returns:
        str or tuple: The generated text. If output_full is True, returns a tuple containing the full generation and the decoded fine prompt.
    """
    # generation with more control
    x_semantic = generate_text_semantic(
        text_prompt,
        history_prompt=voice_name,
        temp=semantic_temp,
        min_eos_p=eos_p,
        use_kv_caching=True
    )

    x_coarse_gen = generate_coarse(
        x_semantic,
        history_prompt=voice_name,
        temp=coarse_temp,
        use_kv_caching=True
    )
    x_fine_gen = generate_fine(
        x_coarse_gen,
        history_prompt=voice_name,
        temp=fine_temp,
    )

    if output_full:
        full_generation = {
            'semantic_prompt': x_semantic,
            'coarse_prompt': x_coarse_gen,
            'fine_prompt': x_fine_gen
        }
        return full_generation, codec_decode(x_fine_gen)
    return codec_decode(x_fine_gen)


def text_to_semantic(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
):
    """Generate semantic array from text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar

    Returns:
        numpy semantic array to be fed into `semantic_to_waveform`
    """
    x_semantic = generate_text_semantic(
        text,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    return x_semantic


def semantic_to_waveform(
    semantic_tokens: np.ndarray,
    history_prompt: Optional[Union[Dict, str]] = None,
    temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from semantic input.

    Args:
        semantic_tokens (np.ndarray): Semantic token output from `text_to_semantic`.
        history_prompt (Optional[Union[Dict, str]], optional): History choice for audio cloning. Defaults to None.
        temp (float, optional): Generation temperature (1.0 more diverse, 0.0 more conservative). Defaults to 0.7.
        silent (bool, optional): Disable progress bar. Defaults to False.
        output_full (bool, optional): Return full generation to be used as a history prompt. Defaults to False.

    Returns:
        np.ndarray: Numpy audio array at sample frequency 24khz.
        
        If `output_full` is True, the function returns a tuple containing the full generation and the audio array:
        - full_generation (dict): A dictionary containing the following prompts:
            - "semantic_prompt": semantic_tokens
            - "coarse_prompt": coarse_tokens
            - "fine_prompt": fine_tokens
        - audio_arr (np.ndarray): Numpy audio array at sample frequency 24khz.
    """
    coarse_tokens = generate_coarse(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=temp,
        silent=silent,
        use_kv_caching=True
    )
    fine_tokens = generate_fine(
        coarse_tokens,
        history_prompt=history_prompt,
        temp=0.5,
    )
    audio_arr = codec_decode(fine_tokens)
    if output_full:
        full_generation = {
            "semantic_prompt": semantic_tokens,
            "coarse_prompt": coarse_tokens,
            "fine_prompt": fine_tokens,
        }
        return full_generation, audio_arr
    return audio_arr


def save_as_prompt(filepath, full_generation):
    assert(filepath.endswith(".npz"))
    assert(isinstance(full_generation, dict))
    assert("semantic_prompt" in full_generation)
    assert("coarse_prompt" in full_generation)
    assert("fine_prompt" in full_generation)
    np.savez(filepath, **full_generation)


def generate_audio(
    text: str,
    history_prompt: Optional[Union[Dict, str]] = None,
    text_temp: float = 0.7,
    waveform_temp: float = 0.7,
    silent: bool = False,
    output_full: bool = False,
):
    """Generate audio array from input text.

    Args:
        text: text to be turned into audio
        history_prompt: history choice for audio cloning
        text_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        waveform_temp: generation temperature (1.0 more diverse, 0.0 more conservative)
        silent: disable progress bar
        output_full: return full generation to be used as a history prompt

    Returns:
        numpy audio array at sample frequency 24khz
    """
    semantic_tokens = text_to_semantic(
        text,
        history_prompt=history_prompt,
        temp=text_temp,
        silent=silent,
    )
    out = semantic_to_waveform(
        semantic_tokens,
        history_prompt=history_prompt,
        temp=waveform_temp,
        silent=silent,
        output_full=output_full,
    )
    if output_full:
        full_generation, audio_arr = out
        return full_generation, audio_arr
    else:
        audio_arr = out
    return audio_arr



