import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()


setuptools.setup(
  name = 'qt_trajectories',
  packages = ['qt_trajectories'], 
  version = '0.1',
  license='MIT',
  description = 'Library focused on simulate quantum trajectories with different unravelings',
  author = 'Diego Veloza Diaz', 
  author_email = 'dvelozad@unal.edu.co',
  url = 'https://github.com/dvelozad',  
  download_url = 'https://github.com/dvelozad/qt-unraveling-python/archive/refs/tags/v_01.tar.gz',    
  keywords = ['python','quantum control', 'unraveling', 'master equation', 'lindblad', 'open systems'],   
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
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
  ],
)