from setuptools import setup

setup(name='pymicrofluidics',
      version='0.2.2',
      description='Making microfludics designs',
      url='https://github.com/guiwitz/PyMicrofluidics',
      author='Guillaume Witz',
      author_email='',
      license='MIT',
      packages=['pymicrofluidics'],
      zip_safe=False,
      package_data = {'pymicrofluidics': ['data/hershey.txt']},
      install_requires=[
            'numpy',
            'dxfwrite',
            'shapely'
      ],)