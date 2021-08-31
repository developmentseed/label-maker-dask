"""Setup for label-maker-dask"""

from setuptools import setup

with open("README.md") as f:
    readme = f.read()

# Runtime requirements.
inst_reqs = [
    "dask",
    "fiona",
    "mercantile",
    "numpy",
    "Pillow",
    "rasterio",
    "requests",
    "rio-cogeo",
]

extra_reqs = {
    "test": ["pytest"],
}

setup(
    name="label-maker-dask",
    version="0.1.0",
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
    zip_safe=False,
    install_requires=inst_reqs,
    extras_require=extra_reqs,
)
