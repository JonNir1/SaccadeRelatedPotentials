from typing import List, Tuple, Union

import easygui_qt.easygui_qt as gui

_MAX_TRIALS = 3
_MAX_TRIALS_ERROR_FORMAT = "Invalid input after %d attempts. Exiting."


def boolean_modal(
        title: str = "Yes or No?",
        msg: str = "Please choose:",
        max_trials: int = _MAX_TRIALS,
) -> bool:
    assert max_trials >= 1
    user_input = None
    cnt = 0
    while user_input is None:
        cnt += 1
        user_input = gui.get_yes_or_no(title=title, message=msg,)
        if isinstance(user_input, bool):
            break
        if cnt > max_trials:
            break
    if user_input is None:
        raise RuntimeError(_MAX_TRIALS_ERROR_FORMAT % max_trials)
    return user_input


def string_modal(
        title: str = "Insert String",
        msg: str = "Please enter a value:",
        default: str = "",
        max_trials: int = _MAX_TRIALS,
) -> str:
    assert max_trials >= 1
    user_input = ""
    cnt = 0
    while not user_input:
        cnt += 1
        user_input = gui.get_string(title=title, message=msg, default_response=default,)
        if isinstance(user_input, str):
            user_input = user_input.strip()
            if user_input:
                break
        if cnt > max_trials:
            break
    if not user_input:
        raise RuntimeError(_MAX_TRIALS_ERROR_FORMAT % max_trials)
    return user_input


def single_choice_modal(
        title: str = "Choose an Option",
        msg: str = "Please choose:",
        choices: Union[List[str], Tuple[str]] = None,
        max_trials: int = _MAX_TRIALS,
) -> str:
    assert max_trials >= 1
    user_input = ""
    cnt = 0
    while not user_input:
        cnt += 1
        user_input = gui.get_choice(title=title, message=msg, choices=choices,)
        if cnt > max_trials:
            break
    if not user_input:
        raise RuntimeError(_MAX_TRIALS_ERROR_FORMAT % max_trials)
    return user_input


def multiple_choices_modal(
        title: str = "Select Options",
        choices: Union[List[str], Tuple[str]] = None,
        max_trials: int = _MAX_TRIALS,
) -> List[str]:
    assert max_trials >= 1
    user_input = None
    cnt = 0
    while user_input is None:
        cnt += 1
        user_input = gui.get_list_of_choices(title=title, choices=choices,)
        if isinstance(user_input, list):
            break
        if cnt > max_trials:
            break
    if user_input is None:
        raise RuntimeError(_MAX_TRIALS_ERROR_FORMAT % max_trials)
    return user_input

