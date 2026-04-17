from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'tauv_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Make sure calibrationFiles and weights ends up in a predictable place
        (os.path.join('share', package_name, 'calibrationFiles'), glob('calibrationFiles/*')),
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
        (os.path.join('share', package_name, 'configs'), glob('configs/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*launch.[pxy][yma]*'))),
    ],
    install_requires=[
        'setuptools',
        'numpy',
        'opencv-python',
        'cv_bridge'
    ],
    zip_safe=True,
    maintainer='Nicolas Frazier',
    maintainer_email='nfrazier@andrew.cmu.edu',
    description='Vision Stack for Tartan AUV',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'cameraPublisher = tauv_vision.cameraPublisher:main'
        ],
    },
)