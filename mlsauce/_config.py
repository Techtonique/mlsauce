"""Global configuration state and functions for management -- adapted from sklearn
"""

import os
from contextlib import contextmanager as contextmanager

_global_config = {
    "assume_finite": bool(os.environ.get("MLSAUCE_ASSUME_FINITE", False)),
    "working_memory": int(os.environ.get("MLSAUCE_WORKING_MEMORY", 1024)),
    "print_changed_only": True,
    "display": "text",
}


def get_config():
    """Retrieve current values for configuration set by :func:`set_config`

    Returns
    -------
    config : dict
        Keys are parameter names that can be passed to :func:`set_config`.

    See Also
    --------
    config_context: Context manager for global mlsauce configuration
    set_config: Set global mlsauce configuration
    """
    return _global_config.copy()


def set_config(
    assume_finite=None,
    working_memory=None,
    print_changed_only=None,
    display=None,
):
    """Set global mlsauce configuration

    .. versionadded:: 0.3.0

    Parameters
    ----------
    assume_finite : bool, optional
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

        .. versionadded:: 0.3.0

    working_memory : int, optional
        If set, mlsauce will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

        .. versionadded:: 0.3.0

    print_changed_only : bool, optional
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()' while the default
        behaviour would be to print 'SVC(C=1.0, cache_size=200, ...)' with
        all the non-changed parameters.

        .. versionadded:: 0.3.0

    display : {'text', 'diagram'}, optional
        If 'diagram', estimators will be displayed as text in a jupyter lab
        of notebook context. If 'text', estimators will be displayed as
        text. Default is 'text'.

        .. versionadded:: 0.3.0

    See Also
    --------
    config_context: Context manager for global mlsauce configuration
    get_config: Retrieve current values of the global configuration
    """
    if assume_finite is not None:
        _global_config["assume_finite"] = assume_finite
    if working_memory is not None:
        _global_config["working_memory"] = working_memory
    if print_changed_only is not None:
        _global_config["print_changed_only"] = print_changed_only
    if display is not None:
        _global_config["display"] = display


@contextmanager
def config_context(**new_config):
    """Context manager for global mlsauce configuration

    Parameters
    ----------
    assume_finite : bool, optional
        If True, validation for finiteness will be skipped,
        saving time, but leading to potential crashes. If
        False, validation for finiteness will be performed,
        avoiding error.  Global default: False.

    working_memory : int, optional
        If set, mlsauce will attempt to limit the size of temporary arrays
        to this number of MiB (per job when parallelised), often saving both
        computation time and memory on expensive operations that can be
        performed in chunks. Global default: 1024.

    print_changed_only : bool, optional
        If True, only the parameters that were set to non-default
        values will be printed when printing an estimator. For example,
        ``print(SVC())`` while True will only print 'SVC()', but would print
        'SVC(C=1.0, cache_size=200, ...)' with all the non-changed parameters
        when False. Default is True.

        .. versionadded:: 0.3.0

    display : {'text', 'diagram'}, optional
        If 'diagram', estimators will be displayed as text in a jupyter lab
        of notebook context. If 'text', estimators will be displayed as
        text. Default is 'text'.

        .. versionadded:: 0.3.0

    Notes
    -----
    All settings, not just those presently modified, will be returned to
    their previous values when the context manager is exited. This is not
    thread-safe.

    Examples
    --------
    >>> import mlsauce
    >>> from mlsauce.utils.validation import assert_all_finite
    >>> with mlsauce.config_context(assume_finite=True):
    ...     assert_all_finite([float('nan')])
    >>> with mlsauce.config_context(assume_finite=True):
    ...     with mlsauce.config_context(assume_finite=False):
    ...         assert_all_finite([float('nan')])
    Traceback (most recent call last):
    ...
    ValueError: Input contains NaN, ...

    See Also
    --------
    set_config: Set global mlsauce configuration
    get_config: Retrieve current values of the global configuration
    """
    old_config = get_config().copy()
    set_config(**new_config)

    try:
        yield
    finally:
        set_config(**old_config)
