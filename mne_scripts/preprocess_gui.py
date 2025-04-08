import os
import time
from typing import Optional, List

import mne
import easygui_qt.easygui_qt as gui

import mne_scripts.helpers.gui_helpers as guih
import mne_scripts.helpers.raw_helpers as rawh
import mne_scripts.helpers.annotation_helpers as annh

_MIN_FREQ, _MAX_FREQ, _NOTCH_FREQ = 0.1, 100, 50
_REF_ELECTRODE = "average"
_VOLTAGE_JUMP_THRESHOLD, _VOLTAGE_JUMP_DURATION_MS, _VOLTAGE_JUMP_MIN_RATIO = 2e-4, 100, 0.5
_BAD_VOLTAGE_PRE_MS, _BAD_VOLTAGE_POST_MS, _BAD_VOLTAGE_MERGE_MS = 250, 250, 50
_VISUALIZATION_CHANNELS, _VISUALIZATION_SCALING = 15, dict(eeg=1e-4, eog=1e-4, eyegaze=5e2, pupil=5e2)



def edit_montage(raw: mne.io.Raw, montage: Optional[str] = None) -> mne.io.Raw:
    """
    Set the montage for the data based on the provided montage or use a GUI to select from MNE's built-in montages.
    If the data already has a montage, the user is prompted whether to overwrite it.

    :param: raw: MNE Raw object
    :param: montage: Optional string specifying the montage to set. If None, a GUI will be used to select a montage.

    :return: MNE Raw object with the new montage set.
    """
    existing_montage = raw.get_montage()
    if existing_montage is not None:
        do_overwrite = guih.boolean_modal(
            title="Overwrite Montage?",
            msg="Do you want to overwrite the object's existing montage?",
        )
        if not do_overwrite:
            return raw
    if montage is not None:
        rawh.set_new_montage(raw, montage, overwrite=True)
        return raw
    # reached here if no montage is provided -> prompt user to select a montage
    all_montages = mne.channels.get_builtin_montages() + ["No Montage"]     # add "No Montage" option
    montage = guih.single_choice_modal(
        title="Choose Montage",
        msg="Select a montage from the list:",
        choices=all_montages,
    )
    montage = None if montage == "No Montage" else montage
    return rawh.set_new_montage(raw, montage, overwrite=True)


def edit_channel_types(
        raw: mne.io.Raw,
        eog_channels: Optional[List[str]] = None,
        stim_channels: Optional[List[str]] = None,
        gaze_channels: Optional[List[str]] = None,
        pupil_channels: Optional[List[str]] = None,
        misc_channels: Optional[List[str]] = None,
) -> mne.io.Raw:
    """
    Edit the channel types of the data, either as EEG, EOG, Gaze, Pupil, Stim, or Misc.
    If the user provides a list of channels for each type (which could be empty), those channels will be set to the
    corresponding type. Otherwise, the user is prompted to select the channels for each type from a list of all channels.
    If the data already has channel types set, the user is prompted whether to overwrite them.

    :param raw: MNE Raw object
    :param eog_channels: List of channels to set as EOG. If None, the user will be prompted to select channels.
    :param stim_channels: List of channels to set as Stim. If None, the user will be prompted to select channels.
    :param gaze_channels: List of channels to set as Gaze. If None, the user will be prompted to select channels.
    :param pupil_channels: List of channels to set as Pupil. If None, the user will be prompted to select channels.
    :param misc_channels: List of channels to set as Misc. If None, the user will be prompted to select channels.

    :return: new MNE Raw object with the new channel types set.
    """
    existing_types = set(raw.get_channel_types())
    if existing_types != {'eeg'}:
        do_remap = guih.boolean_modal(
            title="Overwrite Channel Types?",
            msg="The data already has channel types set. Do you want to overwrite them?",
        )
        if not do_remap:
            return raw

    if (eog_channels is not None) and \
            (stim_channels is not None) and \
            (gaze_channels is not None) and \
            (pupil_channels is not None) and \
            (misc_channels is not None):
        return rawh.remap_channels(
            raw,
            eog_channels=eog_channels,
            stim_channels=stim_channels,
            gaze_channels=gaze_channels,
            pupil_channels=pupil_channels,
            misc_channels=misc_channels,
        )

    # reached here if some channel types aren't provided -> prompt user to select channel types
    gui.show_message(
        title="Set Channel Types",
        message="Choose which channels to set as EOG, Gaze, Pupil, Stim, and Misc.\n" +
                "Remaining channels will be set as EEG.",
    )
    channels = raw.ch_names
    if eog_channels is None:
        eog_channels = guih.multiple_choices_modal(title="EOG Channels", choices=channels)
    eog_channels = list(set(eog_channels) - set(channels))
    if stim_channels is None:
        stim_channels = guih.multiple_choices_modal(title="Stim Channels", choices=channels)
    stim_channels = list(set(stim_channels) - set(channels))
    if gaze_channels is None:
        gaze_channels = guih.multiple_choices_modal(title="Gaze Channels", choices=channels)
    gaze_channels = list(set(gaze_channels) - set(channels))
    if pupil_channels is None:
        pupil_channels = guih.multiple_choices_modal(title="Pupil Channels", choices=channels)
    pupil_channels = list(set(pupil_channels) - set(channels))
    if misc_channels is None:
        misc_channels = guih.multiple_choices_modal(title="Misc Channels", choices=channels)
    misc_channels = list(set(misc_channels) - set(channels))

    if set(eog_channels) & set(stim_channels) & set(gaze_channels) & set(pupil_channels) & set(misc_channels):
        raise ValueError("Channel types cannot overlap.")
    return rawh.remap_channels(
        raw,
        eog_channels=eog_channels,
        stim_channels=stim_channels,
        gaze_channels=gaze_channels,
        pupil_channels=pupil_channels,
        misc_channels=misc_channels,
    )


def edit_sampling_rate(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Edit the sampling rate of the data. The user is prompted to enter a new sampling rate.
    If the user enters an invalid value, they are prompted to enter a new value again.

    :param: raw: MNE Raw object

    :return: MNE Raw object with the new sampling rate set.
    """
    new_sfreq = guih.get_float_modal(
        title="Set Sampling Rate",
        msg="Enter a new sampling rate (Hz):",
        default=raw.info['sfreq'],
        max_trials=3,
    )
    if new_sfreq <= 0:
        raise ValueError("Sampling rate must be greater than 0.")
    raw.resample(new_sfreq, verbose=False)
    return raw

