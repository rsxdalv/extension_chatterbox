
import setuptools
import re
import os

setuptools.setup(
	name="tts_webui_extension.chatterbox",
    packages=setuptools.find_namespace_packages(),
    version="4.1.0",
	author="rsxdalv",
	description="Chatterbox TTS extension for text-to-speech generation.",
	url="https://github.com/rsxdalv/tts_webui_extension.chatterbox",
    project_urls={},
    scripts=[],
    install_requires=[
        "gradio",
        "chatterbox-tts @ git+https://github.com/rsxdalv/chatterbox@faster",
        # "peft",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

