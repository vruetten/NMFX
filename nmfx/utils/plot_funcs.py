from matplotlib import pyplot as pl
import numpy as np

def plot_loss(results_path, losses):
    pl.figure(figsize = (21, 3))
    title = 'loss'
    pl.title(title)
    xax = np.arange(len(losses))
    pl.plot(xax, losses)
    pl.xlabel('iteration #')
    pl.tight_layout()
    pl.savefig(results_path + title + '.png', transparent = True)



      