from dask import delayed
import abc


class Flow(abc.ABC):
    @abc.abstractmethod
    def flow(self):
        pass

    def __call__(self, *args, **kwargs):
        return delayed(self.flow, name=self.__class__.__name__)(*args, **kwargs)
