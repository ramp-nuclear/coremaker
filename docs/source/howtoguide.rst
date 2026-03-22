############
How-To Guide
############
This part of the documentation deals with how users expect to use this package.

When using this package directly, the common reason is for construction of core
elements or a core object itself. We can divide these operations into three
common routines:

#. Material mixture specification
#. Element construction through composition
#. Assembling a core from elements and lattices

In this part of the guide we will cover the different parts of the code you will
be interested in during you work in these routines, and link to examples for each
of those.

*********************
Mixture Specification
*********************
When one wants to model some component, one has to know what it is made of.
Commonly, one would have access to specification of the material by Isotopic
number densities or through a specification of a given alloy in terms of weight
percentage or impurity limits.
Sometimes, material compositions are unknown or guessed at, in which case
standard specifications such as ASME, ANSI or the `PNNL compendium`_ are useful
resources.
In this package we support some general specifications as well, which are
commonplace in certain power and research reactors.

In any such case, one will be interested in the :py:mod:`coremaker.materials`
subpackage. The main object to look for is the :class:`~coremaker.materials.mixture.Mixture`
class.
This class has multiple classmethods for easy construction in common cases, such
as when weight impurities are known or when considering a metallic alloy.
An in-depth specification of a few :class:`~coremaker.materials.mixture.Mixture`
objects can be found in the :doc:`Material Specification`.

*********************
Element Specification
*********************
Most reactor elements are composed of multiple, relatively-homogeneous materials,
that are arranged in some way and connected together, so that they move as one
assembly.
Common elements include fuel bundles and control blades. If we consider a common
PWR fuel bundle, it is made up of multiple pins at regular intervals. The pins
are held in place by spacers at certain heights, and each pin is made up of
multiple stacked UO2 pellets, commonly cladded in Zircalloy-4 cladding.
Some pins in the bundle may include burnable absorbers inside the cladding,
either mixed with the fuel in the pellets or as their own pins (such as in WABA).
Some rods are coated with some neutron absorber such as boron (such as in IFBA).

In neutron transport solvers, it is most common to require that the geometry of
the different homogeneous materials be specified.
It is fairly common to require that each region be specified on its own, with
no direct way to model physical adhesion between different mixtures.
A common way to do so is with CSG definitions, such as those used by many Monte
Carlo codes, such as OpenMC.
These can be difficult to construct directly from the element specification,
and much more difficult to then edit, such as when doing UQ analyses, or when
modelling a proposed change to some element.
That's because a change to one region does not change other related regions.

One cannot simplify the problem beyond its own specification, but a good model
makes the problem complicated rather than complex.
We decided to tackle this problem by basing our implementation of an element
on a Tree structure.
This way, we can encode encapsulation of different objects such as the relations
found between clad and fuel, or between burnup regions within the fuel, by
stating them as graph edges between encapsulating parent nodes and contained
progeny.
An in-depth explanation of the tree structure is given in the 
:ref:`Elements <Elements reference>` section of the reference guide.

Thus, when we model such elements, one has to decompose the object in their head
into basic elements, and model the different relations they expect between the
different regions. Then, one can either construct the tree structure directly
using the :class:`~coremaker.tree.Tree` object, or they can use the tools under
the :py:mod:`coremaker.elements` subpackage for convenience.
The tools therein focus on creating common Tree structures.

Examples of common element constructions can be found in the 
:doc:`Element Examples` notebook.
That notebook shows both manual construction of an element as well as use of
common tools.

Best Practices
==============
The following things are considered best practices when making objects:

#.  Avoid exclusions when possible. Making simpler components makes things usually faster.
#.  Some adapters have a limit for the number of possible components, that are easily
    reached in relevant settings. In those cases, prefer bunching up components that don't
    really have to be split.
#.  Avoid external exclusions unless they are necessary. They are a level of abstraction
    that is relatively deep, and they're only suppoed to be used when there's some sort of
    geometry sticking into another geometry without being encapsulated in it. They void
    volume computations and the like, so they are to be avoided when you can.

******************
Core Specification
******************
Reactor cores are commonly made up of multiple rods that fit within a repeating
lattice, which is supported by a grid. Those are often encased with a neutron
reflector and a structural vessel, followed by a large radiation shield.
It is common for assemblies to be moved and rotated between grid sites during
operation, so we want our model to have a clean and simple way to shift these
assemblies in the grid, while maintaining the same type of implementation as
that used in the `Element Specification`_ section, to reduce the cognitive load
on the researcher modelling the core.
Thus, the core object holds its own Tree, to specifiy everything in the core
that isn't placed in the grid, as well as where the grid fits in.
The grid is its own object, and is made up of one or more lattices, into which
trees can be inserted at different sites.

Grid Specification
==================
Grids can come in multiple different shapes. We support Cartesian and Hexagonal
grids.
Cartesian grids are also available in cases where the grid is made up of multiple
lattices, which are separated by regions where other core components lie.
This is the case, for example, in the OPAL reactor in Australia, which is made
up of 4 lattices separated by a "+" shaped region for control blades.

The grid objects can be found in the :py:mod:`coremaker.grids` subpackage.
Creation of grid objects isn't covered as its own subject, but can be found
in the examples for core creations, given below, or in our repositories for
specific reactor benchmarks, such as the OPAL reactor, which we plan to publish
alongside the rest of the RAMP repositories under `ramp-nuclear` on GitHub.
When they are available, we will probably remember to reference them here.

.. TODO! Try to remember to put links to OPAL when it is up and running.

Core Specification
==================
Full core specification examples can be found in :doc:`Core Examples` notebook.


.. _PNNL compendium: https://www.pnnl.gov/main/publications/external/technical_reports/PNNL-15870Rev2.pdf
