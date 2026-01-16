from setuptools import find_packages, setup
from typing import List

hypen_e = '-e .'


def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    with open(file_path) as file_obj:
        requirements = [
            req.strip()
            for req in file_obj.readlines()
            if req.strip() and req.strip() != "-e ."
        ]
    return requirements


setup(
    name = 'mlproject',
    version= '0.0.1',
    author='Atharv',
    author_email='atharvakhandelwal19@gmail.com',
    packages= find_packages(),
    install_requires = get_requirements('requirements.txt')
)
