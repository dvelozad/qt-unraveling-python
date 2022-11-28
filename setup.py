import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="qt_unraveling",
    version="0.2.4",
    author="Diego Veloza Diaz",
    author_email="dvelozad@unal.edu.co",
    description="Library focused on simulate quantum trajectories with different unravelings",
    url="https://github.com/dvelozad/qt-unraveling-python",
    keywords = ['python','quantum control', 'unraveling', 'master equation', 'lindblad', 'open systems'],
    download_url = 'https://github.com/dvelozad/qt-unraveling-python/archive/refs/tags/v0.2.4.tar.gz',   
    packages=['qt_unraveling'],
    install_requires=[            # I get to this in a second
          'numpy',
          'numba',
          'scipy',
          'matplotlib'
      ],
    classifiers=[
        'Development Status :: 3 - Alpha',      
        'Intended Audience :: Science/Research',      
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',   
        'Programming Language :: Python :: 3',     
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
