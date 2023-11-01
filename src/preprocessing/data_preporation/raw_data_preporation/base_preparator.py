from pathlib import Path
from abc import ABC, abstractmethod

class BasePrepare(ABC):

    def __init__(self, raw_path: Path, save_path: Path) -> None:
        self.raw_path = raw_path
        self.save_path = save_path
    
    @abstractmethod
    def make_dataset(self) -> None:
        raise NotImplementedError
