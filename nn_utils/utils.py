# General imports
import os
import torch

# Project specific imports

# Imports from internal libraries

# Typing imports
from typing import TYPE_CHECKING


# if TYPE_CHECKING:

def select_device(ngpu=1):
    device = torch.device("cuda:0" if (
            torch.cuda.is_available() and ngpu > 0) else "cpu")
    return device


class MetaInt(int):
    def __new__(cls, value, name=None):
        i = int.__new__(cls, value)
        return i

    def __init__(self, value, name=None):
        int.__init__(value)
        self.name = name


class MetaFloat(float):
    def __new__(cls, value, name=None):
        i = float.__new__(cls, value)
        return i

    def __init__(self, value, name=None):
        float.__init__(value)
        self.name = name


if __name__ == '__main__':
    print(f'Running {__file__}')
    print(f"Script dir:  {os.path.dirname(os.path.abspath(__file__))}")
    print(f"Working dir: {os.path.abspath(os.getcwd())}")
