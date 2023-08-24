.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/sonecules.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/sonecules
    .. image:: https://readthedocs.org/projects/sonecules/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://sonecules.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/sonecules/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/sonecules
    .. image:: https://img.shields.io/pypi/v/sonecules.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/sonecules/
    .. image:: https://img.shields.io/conda/vn/conda-forge/sonecules.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/sonecules
    .. image:: https://pepy.tech/badge/sonecules/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/sonecules
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/sonecules

.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white
   :target: https://github.com/pre-commit/pre-commit
   :alt: pre-commit

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: black

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

=========
sonecules
=========


    a Python Sonification Architecture


sonecules are sonification designs wrapped in a concise object-oriented interface based on `mesonic <https://github.com/interactive-sonification/mesonic/>`_.


.. _pyscaffold-notes:

Making Changes & Contributing
=============================

This project uses `pre-commit`_, please make sure to install it before making any
changes::

    pip install pre-commit
    cd sonecules
    pre-commit install

It is a good idea to update the hooks to the latest version::

    pre-commit autoupdate

Don't forget to tell your contributors to also install and use pre-commit.

.. _pre-commit: https://pre-commit.com/


How to cite sonecules
=====================

Authors of scientific papers using sonecules are encouraged to cite the following paper.

.. code-block:: none

    @inproceedings{ReinschHermann-ICAD2023-Sonecules,
    author       = {Reinsch, Dennis and Hermann, Thomas},
    location     = {Norrköping, Sweden},
    publisher    = {ICAD},
    title        = {{sonecules: a Python Sonification Architecture}},
    url          = {https://pub.uni-bielefeld.de/record/2979095},
    year         = {2023},
    }


Note
====

This project has been set up using PyScaffold 4.4. For details and usage
information on PyScaffold see https://pyscaffold.org/.
