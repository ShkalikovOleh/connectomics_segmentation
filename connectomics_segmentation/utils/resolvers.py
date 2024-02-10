from functools import partial, reduce

from omegaconf import OmegaConf


def register_custom_resolvers() -> None:
    OmegaConf.register_new_resolver(
        "prod_list", partial(reduce, lambda x, y: x * y)  # type: ignore
    )
    OmegaConf.register_new_resolver("ipow", lambda x, y: int(x**y))
