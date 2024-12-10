from setuptools import setup, find_packages

setup(
    name="NN_Profiles",
    version="0.1.0",
    description="A description of your suite",
    author="Isaac Malsky",
    author_email="isaacmalsky@gmail.com",
    url="https://github.com/imalsky/NN_Profiles",
    packages=find_packages(),
    install_requires=[
        "exo_k==1.3.0",
        "numpy==2.2.0",
        "torch==2.5.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
