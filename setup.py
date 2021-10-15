"""Setup for label-maker-dask"""

from setuptools import find_packages, setup

with open("README.md") as f:
    readme = f.read()

# Runtime requirements.
inst_reqs = [
    "dask",
    "mapbox-vector-tile",
    "mercantile",
    "numpy",
    "Pillow",
    "rasterio",
    "requests",
    "rio-tiler>=2",
    "shapely",
]

extra_reqs = {
    "test": ["pytest"],
}

setup(
    name="label-maker-dask",
    version="0.1.1",
    python_requires=">=3.6",
    description="Run label maker as a dask job",
    long_description=readme,
    long_description_content_type="text/markdown",
    classifiers=[
        "Intended Audience :: Information Technology",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    keywords="",
    author="Drew Bollinger",
    author_email="drew@developmentseed.org",
    url="https://github.com/developmentseed/label-maker-dask",
    license="BSD",
    packages=find_packages(exclude=["tests*"]),
    zip_safe=False,
    install_requires=inst_reqs,
    extras_require=extra_reqs,
)
