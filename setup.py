import setuptools

setuptools.setup(
    name='keras-mixed-scale-dense-net',
    version='0.0.1',
    url='https://github.com/efornaciari/keras-mixed-scale-dense-net',
    author='Eric Fornaciari',
    maintainer='Eric Fornaciari',
    packages=setuptools.find_packages(),
    install_requires=['keras', 'keras-resnet', 'six', 'scipy'],
)