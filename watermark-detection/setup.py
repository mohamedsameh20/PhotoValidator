from setuptools import setup, find_packages

setup(
    name="wmdetection",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch",
        "torchvision", 
        "pillow",
        "numpy",
        "matplotlib",
        "tqdm",
        "huggingface-hub",
        "opencv-python",
        "timm>=0.6.12",
        "pandas==2.2.2",
        "scikit_learn==1.5.0"
    ],
    python_requires=">=3.7",
    description="Watermark detection using CNN models",
    author="boomb0om",
)
