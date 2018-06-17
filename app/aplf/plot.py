

def plot(y,
         x,
         path,
         **kwargs):
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(x, y)
    plt.savefig(path)
    plt.close()
    return path
