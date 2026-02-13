# Core Maker
The CoreMaker package is a nuclear reactor core modelling package.
This package aims to make it easy for researchers to create reproducible, interoperable
descriptions of nuclear reactor cores.

This package is developed as part of the Reactor Analysis Management Program project,
where it is used for defining the concepts of a nuclear core and implementations of
various geometries and materials.

To understand the core concepts of this package, please check out our documentation.
Building the documentation is done by writing "make html" in the `docs` folder once
all the requirements are installed.

## How to Contribute
Our documentation is probably where we have the most to gain from new contributors.
Please feel free to read through it, ask questions and help us make it a better tool for
beginners.

Users who want support for additional `Surfaces`, `Geometries` or `Constructions`
are welcome to suggest and contribute new models.
If you can send us the use case for core components where such tools are necessary,
we will gladly consider your contribution.

Another relatively good place to start is with writing tests.
Our test coverage could still be better, and writing tests is a good way to make yourself
familir with our API.

For more experienced developers, one of our main next goals is to support a human readable
serialization of our models.
