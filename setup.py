import os
from glob import glob

from setuptools import setup

package_name = 'urdf_to_dh'

# build a list of the data files
data_files = []
data_files.append(("share/ament_index/resource_index/packages", ["resource/" + package_name]))
data_files.append(("share/" + package_name, ["package.xml"]))

def package_files(directory, data_files):
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            data_files.append(("share/" + package_name + "/" + path, glob(path + "/**/*.*", recursive=True)))
    return data_files

data_files = package_files('launch/', data_files)
data_files = package_files('urdf/', data_files)

setup(
    name=package_name,
    version='0.0.2',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='andy and Takumi Asada',
    maintainer_email='andy.mcevoy@sslmda.com',
    description='Generate DH parameters from a URDF',
    license='Apache 2',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'generate_dh = urdf_to_dh.generate_dh:main'
        ],
    },
)
