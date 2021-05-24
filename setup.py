from distutils.core import setup
setup(
  name = 'qt_trajectories',         # How you named your package folder (MyLib)
  packages = ['qt_trajectories'],   # Chose the same as "name"
  version = '0.1',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Library focused on simulate quantum trajectories with different unravelings',   # Give a short description about your library
  author = 'Diego Veloza Diaz',                   # Type in your name
  author_email = 'dvelozad@unal.edu.co',      # Type in your E-Mail
  url = 'https://github.com/dvelozad',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/dvelozad/qt-unraveling-python/archive/refs/tags/v_01.tar.gz',    # I explain this later on
  keywords = ['python','quantum control', 'unraveling', 'master equation', 'lindblad', 'open systems'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Physicists',      # Define that your audience are developers
    'Topic :: Scientific Research',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)