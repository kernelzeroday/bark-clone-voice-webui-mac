from typing import Dict, Optional, Union

import numpy as np
from .generation import codec_decode, generate_coarse, generate_fine



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
