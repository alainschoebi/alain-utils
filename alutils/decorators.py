# Python
import functools
import importlib.util

# Logging
from .loggers import get_logger
logger = get_logger(__name__)

available_pkgs_cache = []

def requires_package(*pkgs: str):
    """
    Decorator to check if the required packages are installed before running a
    function. If a package is not installed, an ImportError is raised.

    Args:
        pkgs: one or multiple strings, where each string is the name of a
              package that is required for running the function.
    """
    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for pkg in pkgs:

                if pkg in available_pkgs_cache:
                    continue

                if not importlib.util.find_spec(pkg):
                    logger.error(
                        f"The package '{pkg}' is required for running " +
                        f"function '{function.__qualname__}' and could not " +
                        f"be found."
                    )
                    raise ImportError(
                        f"The package '{pkg}' is required for running " +
                        f"function '{function.__qualname__}' and could not " +
                        f"be found."
                    )

                available_pkgs_cache.append(pkg)
            return function(*args, **kwargs)

        return wrapper

    return decorator

"""
def verify_input_types(*types: type):
    Decorator to check if the input types of a function are correct. If the
    types are not correct, a `TypeError` is raised.

    Args:
        types: `type` one or multiple types, where each type is the expected
               type of an input argument.

    def decorator(function):

        @functools.wraps(function)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, types)):
                if not isinstance(arg, expected_type):
                    logger.error(
                        f"Argument {i} of function '{function.__qualname__}' " +
                        f"should be of type '{expected_type}', but got " +
                        f"type '{type(arg)}' instead."
                    )
                    raise TypeError(
                        f"Argument {i} of function '{function.__qualname__}' " +
                        f"should be of type '{expected_type}', but got " +
                        f"type '{type(arg)}' instead."
                    )

            return function(*args, **kwargs)

        return wrapper

    return decorator
"""
