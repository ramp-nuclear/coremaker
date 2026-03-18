from typing import Any, Type

from ramp_core.serializable import Serializable, deserialize_default

from coremaker.protocols.element import Element
from coremaker.protocols.grid import Site
from coremaker.tree import Tree


def serialize_contents(d: dict[Site, Element]) -> dict[Site, tuple[str, dict[str, Any]]]:
    """Serialize a dictionary of grid contents"""
    return {site: elem.serialize() for site, elem in d.items()}


def deserialize_contents(
    d: dict[Site, tuple[str, dict[str, Any]]], supported: dict[str, Type[Serializable]]
) -> dict[Site, Element]:
    """Deserialize a dictionary of grid contents

    Parameters
    ----------
    d:
        Serialization, usually from a serialize_contents call
    supported: dict[str, Type[Serializable]]
        Mapping of identifiers to python types

    Returns
    -------
    dict[Site, Element]
        A dictionary of what the grid holds, which was previously serialized

    """

    # Safe because we assume people gave us elements to begin with
    # noinspection PyTypeChecker
    return {site: deserialize_default(v, supported=supported, default=Tree) for site, v in d.items()}
