#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

requirements = [
    'numpy',
    'scipy',
    'tabulate',
    'sklearn',
    'nltk',
    'gensim'
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='earthquakeclassifier',
    version='0.1.0',
    description="Classifier for spanish earthquake relevant messages",
    long_description=readme + '\n\n' + history,
    author="Alfredo Cobo",
    author_email='ajscobo@gmail.com',
    url='https://github.com/ajcobo/earthquakeclassifier',
    packages=[
        'earthquakeclassifier',
        'src',
        'earthquakeclassifier/modules'
    ],
    package_dir={'earthquakeclassifier':'earthquakeclassifier',
                 'modules':'earhquakeclassifier/earthquakeclassifier'},
    include_package_data=True,
    install_requires=requirements,
    license="ISCL",
    zip_safe=False,
    keywords='earthquakeclassifier',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: ISC License (ISCL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    
)
