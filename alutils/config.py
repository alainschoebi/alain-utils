# Logging
from .loggers import get_logger
logger = get_logger(__name__)

def update_config_dict(config_to_update: dict,
                       config_to_add: dict,
                       default_config: dict) -> None:
    """
    Set partial configuration.

    Inputs
    - config_to_edit: `dict` base configuration which will be updated
    - config_to_add:  `dict` partial configuration to add
    - default_config: `dict` default configuration used for checking the
                       types and names of the arguments.

    Note: the values of the `config_to_add` configuration are not being copied.
          That is, the `config_to_update` will contain the same objects.
    """

    if not isinstance(config_to_update, dict) or \
       not isinstance(config_to_add, dict) or \
       not isinstance(default_config, dict):
       logger.error("The configurations must be dictionaries.")
       raise ValueError("The configurations must be dictionaries.")

    for key, value in config_to_add.items():

        if not key in default_config:
            logger.error(f"Argument '{key}' not found in the default " +
                         f"configuration.")
            raise ValueError(f"Argument '{key}' not found in the default "  +
                             f"configuration.")

        if not isinstance(value, dict):
            default_type = type(default_config[key])

            # Same type
            if isinstance(value, default_type):
                config_to_update[key] = value

            # Try to convert the value to the default type
            else:
                try:
                    config_to_update[key] = default_type(value)
                except:
                    logger.error(f"Argument '{key}' has type {default_type} " +
                                 f"but the provided value has type " +
                                 f"{type(value)}.")
                    raise ValueError(f"Argument '{key}' has type " +
                                     f"{default_type} but the provided " +
                                     f"value has type {type(value)}.")

        else:
            update_config_dict(
                config_to_update[key], value, default_config[key]
            )
