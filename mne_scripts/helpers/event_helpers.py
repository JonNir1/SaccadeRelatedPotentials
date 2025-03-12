from typing import Union, List

import numpy as np
import pandas as pd
import mne


def extract_events(
        raw: mne.io.Raw, channel: Union[str, List[str]], output: str = "onset", shortest_event: int = 1
) -> np.ndarray:
    """
    Extract events from one or more stimulus channels in a Raw object.

    :param raw: The MNE Raw object containing stim channel data.
    :param channel: A single stim channel name, a list of names, or "all" to use all stim channels.
    :param output: The type of event boundary to return. One of "onset", "offset", or "step". Default is "onset".
    :param shortest_event: Minimum number of samples for an event to be considered valid. Default is 1.

    :returns: An MNE-style event array (n_events × 3) with sample, previous value, and new value.
    """
    output = output.strip().lower()
    assert output in ["onset", "offset", "step"], "Invalid output type. Must be 'onset', 'offset', or 'step'."
    assert shortest_event > 0, "shortest_event must be non-negative"
    all_channel_names = np.array(raw.ch_names)
    is_stim_channel = np.array(raw.get_channel_types()) == "stim"
    stim_channel_names = (all_channel_names[is_stim_channel]).tolist()
    if isinstance(channel, str):
        channel = channel.strip()
        if channel.lower() == "all":
            return mne.find_events(
                raw,
                stim_channel=stim_channel_names,
                output=output,
                shortest_event=shortest_event,
                consecutive=True,
                verbose=False
            )
        channel = [channel]
    unknown_channels = set(channel) - set(stim_channel_names)
    if unknown_channels:
        raise ValueError(f"Unknown stim channel(s): {unknown_channels}")
    return mne.find_events(
        raw,
        stim_channel=channel,
        output=output,
        shortest_event=shortest_event,
        consecutive=True,
        verbose=False
    )
