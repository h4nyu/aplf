from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment()
ex.observers.append(MongoObserver.create(url='mongo:27017',
                                         db_name='sacred'))


@ex.automain
def my_main():
    print('Hello world!')
