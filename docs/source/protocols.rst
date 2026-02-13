Protocols for other packages
============================
If you're developing another package and you want to use core objects or the stuff they bring,
we have conveniently supplied a number of Protocols that you should rely on. If you
rely only on these protocols, you will likely be compatible with other packages in the
ecosystem. Some packages, such as transport code adapters, would require more than just
these general protocols, because they care about concrete implementation details. In
that case, see :ref:`Geometries`.

.. toctree::
   :maxdepth: 2

   protocols.geometrical
   protocols.coreobjects
   protocols.materials