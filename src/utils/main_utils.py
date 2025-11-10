import yaml
from src.exception.exception import HotelReservationException
import os
import sys
import pickle
import numpy as np
from sklearn.metrics import r2_score, accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score




def read_yaml_file(file_path: str) -> dict:
    """
    Read YAML file and return content as dictionary
    
    Args:
        file_path: Path to YAML file
        
    Returns:
        dict: Content of YAML file
    """
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise HotelReservationException(e, sys)
        

def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """
    Write content to YAML file
    
    Args:
        file_path: Path to YAML file
        content: Content to write
        replace: Whether to replace existing file
    """
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise HotelReservationException(e, sys)


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """
    Save numpy array to file
    
    Args:
        file_path: Path to save file
        array: Numpy array to save
    """
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
    except Exception as e:
        raise HotelReservationException(e, sys)


def load_numpy_array_data(file_path: str) -> np.array:
    """
    Load numpy array data from file
    """
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise HotelReservationException(e, sys)

    
def save_object(file_path: str, obj: object) -> None:
    """
    Save Python object to file using pickle
    
    Args:
        file_path: Path to save file
        obj: Object to save
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise HotelReservationException(e, sys)


def load_object(file_path: str) -> object:
    """
    Load object from file path
    """
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} does not exist")
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise HotelReservationException(e, sys)

