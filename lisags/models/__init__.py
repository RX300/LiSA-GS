"""Construction and CLI fields for the final LiSA transport.

The saved representation name remains light_atlas; checkpoint tensor keys and
constructor order are the original method's, so no conversion is needed.
"""
from .light_atlas import LightAtlasTransport


def resolve_config(config):
    return {**LightAtlasTransport.defaults, **config, "geometry": "3dgs"}


def build_transport(config, light_scale):
    config = resolve_config(config)
    return LightAtlasTransport(**{key: config[key] for key in LightAtlasTransport.defaults},
                               light_scale=light_scale)


def add_method_arguments(parser, argv=None):
    parser.add_argument("--representation", choices=["light_atlas"], default="light_atlas")
    group = parser.add_argument_group("LiSA parameters")
    for field in LightAtlasTransport.cli_fields:
        default = LightAtlasTransport.defaults[field]
        group.add_argument("--" + field.replace("_", "-"), type=type(default), default=default)
