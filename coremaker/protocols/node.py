from typing import Protocol

from coremaker.protocols.geometry import Geometry
from coremaker.protocols.mixture import Mixture
from coremaker.transform import Transform


class NodeLike(Protocol):
    """The information one has to include to be a Node for all intents and purposes.

    """
    transform: Transform
    mixture: Mixture | None

    @property
    def geometry(self) -> Geometry:
        """Gets the geometry of the nodelike.

        """
        raise NotImplementedError("This is not implemented on the protocol.")
