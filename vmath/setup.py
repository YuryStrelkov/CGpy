from setuptools import find_packages, setup
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '0.1.1'
DESCRIPTION = 'Vector mathematics for working with three-dimensional and two-dimensional space.'
LONG_DESCRIPTION = 'A set of vector mathematics tools, consisting of 2d and 3d vectors, 3d and 4d' \
                   ' matrices, quaternions, etc....'

# Setting up
setup(
    name="cgeo",
    version=VERSION,
    author="YuryStrelkov (Yury Strelkov)",
    author_email="<ghost_strelkov@mail.ru>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    install_requires=['numpy', 'pillow'],
    keywords=['python', 'geometry', 'vector 2d', 'vector 3d', 'matrix 3x3', 'matrix 4x4',
              'quaternion', 'bezier curve 2d', 'bezier curve 3d', 'bezier patch',
              'quads marching', 'bounding box', 'bounding rect', 'transform 2d', 'transform 3d'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ])
