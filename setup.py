from setuptools import setup

setup(
    name="geoRpro",
    version="0.1",
    description="Library to process geo raster data",
    url="https://github.com/ESS-uzh/geoRpro.git",
    author="Diego Villamaina",
    author_email="diego.villamaina@geo.uzh.ch",
    license="MIT",
    packages=["geoRpro"],
    entry_points={
        "console_scripts": [
            "geoRpro=geoRpro.cli:main"
        ]  # so this directly refers to a function available in workflow.py
    },
    zip_safe=False,
)
