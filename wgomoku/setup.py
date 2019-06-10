import setuptools

long_description = "A collection of Gomoku heuristics"

setuptools.setup(
    name="wgomoku",
    version="0.0.1",
    author="Wolfgang Giersche",
    author_email="wgiersche@gmail.com",
    description="Gomoku Heuristics",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smurve/DeepGomoku",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)