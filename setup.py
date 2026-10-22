## With the help of 'setup.py' i'll be able to build my entire
# ML Application as a package. And even deploying 'python.pypi'.
# and from there anybody can do the installation and can also use it. 



from setuptools import find_packages,setup
from typing import List

DASH_E_DOT='-e .'
def get_requirements(file_path:str)->List[str]:
    '''
        This function will return the list of requirements.
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements]

        if DASH_E_DOT in requirements:
            requirements.remove(DASH_E_DOT)

            
setup(

name='mlproject',
version='0.0.1',
author='Saifuddin',
author_email='mds744007@gmail.com',
packages=find_packages(),
install_requires=get_requirements('requirements.txt')

)