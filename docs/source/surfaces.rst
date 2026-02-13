Concrete Surfaces
=================
Surfaces are concrete implementations of the :class:`~.Surface`
protocol. However, many other packages would need to access these concrete
implementations directly, as the Surface protocol does not tell an adapter how
to access the numeric representation that these surfaces have, so any adapter
that wants to place these surfaces in their geometry needs to access the concrete
type.

Plane
-----

.. autoclass:: coremaker.surfaces.plane.Plane
    :members: __eq__, __neg__, transform
    :undoc-members:
    :special-members: +__eq__, __neg__

Sphere
------
.. autoclass:: coremaker.surfaces.sphere.Sphere
    :members: __eq__, __neg__, transform
    :undoc-members:
    :special-members: +__eq__, __neg__

Cylinder
--------
.. autoclass:: coremaker.surfaces.cylinder.Cylinder
    :members: __eq__, __neg__, transform
    :undoc-members:
    :special-members: +__eq__, __neg__
