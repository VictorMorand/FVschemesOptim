from setuptools import setup

setup(name='trafsyn',
      version='0.1',
      description='Codebase for "Deep learning of first-order nonlinear hyperbolic conservation law solvers"',
      url='https://github.com/VictorMorand/FVschemesOptim.git',
      author='Victor Morand and Nils Müller',
      license='MIT',
      packages=['trafsyn'],
      install_requires = ['numpy','torch', 'matplotlib','imageio', 'pyfunctionbases', "scipy", "tqdm", "pandas"],
      zip_safe=False,
      package_data={'': ['1wave.npy','5waves.npy','LaxHopfHD.npy','LWRexact.npy','Triangular.npy']},
      include_package_data=True)

