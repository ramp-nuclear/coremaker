.. CoreMaker documentation master file.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to CoreMaker's documentation!
=====================================
This package is used to model reactor core components and reactor cores.
These are made up of geometrical objects, each of which is composed of some
material mixture.
Thus, this package contains both tools for the definition of material mixtures
and geometries.
Both geometries and material mixtures follow a Protocol, because we want to
make it easier for other packages to be interoperable with our own, in case
they want a different implementation. We provide our own implementation for
geometries and material mixtures as well.

To make the creation of complex elements more composable, we implement another
Protocol for complex elements, which can be iterated over to get all of the
sub-components that make up the composed element.
Our specific implementation of that Protocol uses a Tree for composing regions
together.

This documentation provides explanations of both the Protocols and the implementations.
Users may also find the "How-To" guide useful if they know what they want to do
and want to find the right part in the software to look it up.

The tests, as well as the `examples` module, provide examples for some models
which we define here, which may inspire you when you want to model some element
of your own.


Installation
------------
Installation is done through pip (as a PyPI package) or as a conda package,
which is hopefully online as well now.


.. toctree::
   :hidden:
   :maxdepth: 3

   howtoguide
   protocols
   reference
   tests

* :ref:`genindex`
* :ref:`modindex`
