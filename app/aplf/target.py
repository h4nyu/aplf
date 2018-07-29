from abc import ABC, abstructmethod

class Target(ABC):

    @abstructmethod
    def exists(self):
        pass
