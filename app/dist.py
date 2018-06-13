
from dask.distributed import Client
client = Client('dask_scheduler:8786')  # set up local cluster on your laptop
print(client)


def inc(x):
    return x + 1


x = client.submit(inc, 10)
L = client.map(inc, range(1000))
