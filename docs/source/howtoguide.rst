How-To Guide
============
This part of the documentation deals with how users expect to use this package.

Installation
------------
To install, in a relevant environment simply run::

    pip install .

in the parent directory. This will cause an installation through conda/mamba of the
relevant
dependencies. This method may take some time. Alternatively, you could do the conda/mamba
installation first, and then do the pip installation::

    mamba env update -f requirements.yml
    pip install .

To install the test and documentation dependencies as well, you could then use mamba
again::

    mamba env update -f docs/conda_requirements.yml
    mamba env update -f test_requirements.yml

Usage
-----
When using this package directly, the common reason is for construction of core
elements or a core object itself. Core elements are most frequently created through the
use of constructions, which are factory functions designed to create new
:class:`~coremaker.tree.Tree` objects, which are the element structures themselves.
The user then creates a core with a grid object, see :class:`~coremaker.grid.Grid`, where
some sites are populated by elements created by the factories, and some ex-grid
elements are defined similarly with their own tree.

Best Practices
--------------
The following things are considered best practices when making objects:

#.  Avoid exclusions when possible. Making simpler components makes things usually faster.
#.  Some adapters have a limit for the number of possible components, that are easily
    reached in relevant settings. In those cases, prefer bunching up components that don't
    really have to be split.
#.  Avoid external exclusions unless they are necessary. They are a level of abstraction
    that is relatively deep, and they're only suppoed to be used when there's some sort of
    geometry sticking into another geometry without being encapsulated in it. They void
    volume computations and the like, so they are to be avoided when you can.

Running the tests
-----------------
To run the tests, simply run the following command::

    pytest -vvv --doctest-modules

This will create a human readable report for every test.

Building the documentation
--------------------------
To make the documentation, at the ``docs`` directory run the following command::

    make html

This will make an html version of the docs under the build directory. The ``index.html``
file in that directory is the main page that should be opened by a browser.

