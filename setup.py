from setuptools import setup, find_packages

setup(
    name="MaMAI",
    version="2.4",
    packages=find_packages(),
    install_requires=[
        'langchain',
        'flask',
        'bs4',
        'urllib3'
    ],
    author="Francesco Bellifemine",
    author_email="effebi.co@gmail.com",
    description="MaMa is a AI wrapper for different tasks.",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/cyberbionik/MaMa",
)
