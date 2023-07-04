#! /usr/bin/python3
# -*- coding: utf-8 -*-
# @FileName: setup.py
# @Author: Li Chengxin 
# @Time: 2023/7/4 13:43

import setuptools

with open('readme.md', 'r') as f:
    long_description = f.read()

with open('requirements.txt', 'r') as fr:
    pkg_requirements = fr.read().split('\n')
    pkg_requirements.remove('')


VERSION = '0.0.3'

setuptools.setup(
    name='SC-Track',
    author="Li Chengxin",
    author_email="914814442@qq.com",
    url="https://github.com/frozenleaves/SC-Track",
    version=VERSION,
    description='single cell tracking package',
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=pkg_requirements,
    entry_points={
        'console_scripts': [
            'sctrack = SCTrack.sctrack:main',
        ],
    },
)
