from setuptools import setup

setup(name='trafsyn',
      version='0.1',
      description='A Python package for synthesizing traffic models.',
      url='http://github.com/NiMlr/traffic-model-synthesis',
      author='Nils Müller',
      license='MIT',
      packages=['trafsyn', 'trafsyn.test', 'trafsyn.data','trafsyn.Baselines'],
      install_requires = ['numpy', 'matplotlib','imageio', 'pyfunctionbases'],
      zip_safe=False,
      package_data={'': ['testdata.npy']},
      include_package_data=True)

