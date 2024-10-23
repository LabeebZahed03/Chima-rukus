# setup.py

from setuptools import setup, find_packages

setup(
    name='Rufus',
    version='0.1.0',
    author='Labeeb Zahed',
    author_email='labeebzahed@gmail.com',
    description='Rufus: Intelligent Web Data Preparation for RAG Agents',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/LabeebZahed03/Chima-rufus',
    packages=find_packages(),
    install_requires=[
        'beautifulsoup4>=4.9.3',
        'requests>=2.25.1',
        'spacy>=3.0.0',
        'en-core-web-sm>=3.0.0',
        'selenium>=4.0.0',
        'webdriver-manager>=3.5.0',
    ],
    python_requires='>=3.7',
)