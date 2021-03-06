from setuptools import setup, find_packages

requires = [
    "toolz",
    "torchvision",
    "torch",
    "pandas",
    "scikit-learn",
    "seaborn",
    "lightgbm",
    "mlboard-client==0.0.9.rc1",
    "scikit-image",
    "opencv-python",
    "Augmentor",
]
dev_requires = [
    "requests",
    "pytest",
    "pytest-mock",
    "pytest-cov",
    "faker",
    "flake8",
    "autopep8",
    "pytest-asyncio",
    "profilehooks",
    "mypy",
    "pytest-mypy",
    "typing_extensions",
]

setup(
    name="aplf",
    version="0.0.0",
    description="TODO",
    author='Xinyuan Yao',
    author_email='yao.ntno@google.com',
    license="TODO",
    packages=find_packages(),
    install_requires=requires,
    extras_require={
        'dev': dev_requires
    }
)
