from venv import create
import os
from subprocess import run

def check_requirements_file():
    """_summary_
    Check if the requirements file exists in the current working directory
    Returns:
        _type_: bool
        _description_: if the requirements file exists, return True, else return False
    """
    if os.path.exists(os.path.join(os.getcwd(), "installation\\requirements.txt")):
        return True
    else:
        return False
    
def check_virtual_environment():
    """_summary_
    Check if the virtual environment exists in the current working directory
    Returns:
        _type_: bool
        _description_: if the virtual environment exists, return True, else return False
    """
    if os.path.exists("yolo8_venv"):
        return True
    else:
        return False

def get_virtual_environment_path():
    """_summary_
    Get the path to the virtual environment
    Returns:
        _type_: str
        _description_: the path to the virtual environment
    """
    return os.path.join(os.getcwd(), "yolo8_venv")

def create_virtual_environment():
    """_summary_
    Create a virtual environment in the current working directory
    Returns:
        _type_: bool
        _description_: if the virtual environment was created, return True, else return False
    """
    venv_path = get_virtual_environment_path()
    create(venv_path, with_pip=True)
    # check if the virtual environment was created
    return check_virtual_environment()

def install_requirements(venv_path):
    """_summary_
    Install the requirements file in the virtual environment
    """
    # check if in the virtual environment
    run([os.path.join(venv_path, "Scripts", "python.exe"), "-m","pip","install", "-r", "installation\\requirements.txt"])

if __name__ == "__main__":
    # Check if the requirements file exists
    if check_requirements_file():
        print("Requirements file found")
    else:
        print("Requirements file not found")
        os.sys.exit(1)
    # Check if the virtual environment exists
    if check_virtual_environment():
        print("Virtual environment already exists, installing requirements")
    else:
        print("Virtual environment not found, creating virtual environment")
        if create_virtual_environment():
            print("Virtual environment created")
        else:
            print("Virtual environment not created")
            os.sys.exit(1)
    venv_path = get_virtual_environment_path()
    # Install the requirements
    install_requirements(venv_path)
    print("Requirements installed")
    os.sys.exit(0)