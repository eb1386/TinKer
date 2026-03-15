from setuptools import setup, find_packages

setup(
    name="tinker",
    version="1.0.0",
    description="Inference optimization framework for small language models on consumer GPUs",
    author="Evan Borodow",
    url="https://github.com/eb1386/TinKer",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.0",
        "pyyaml",
    ],
    extras_require={
        "triton": ["triton>=2.1"],
        "dev": ["pytest", "matplotlib"],
    },
    package_data={
        "": ["configs/*.yaml"],
    },
)
