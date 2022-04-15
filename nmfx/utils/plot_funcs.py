from matplotlib import pyplot as pl
import numpy as np

def plot_loss(results_path, losses):
    pl.figure(figsize = (21, 4))
    title = 'loss'
    pl.title(title)
    xax = np.arange(losses[0].shape[0])
    pl.plot(xax, losses[0],'o-')
    pl.xlabel('iteration #')
    pl.tight_layout()
    pl.savefig(results_path + title + '.png', transparent = True)

    losses = losses.flatten()
    pl.figure(figsize = (21, 4))
    title = 'all loss'
    pl.title(title)
    xax = np.arange(len(losses))
    pl.plot(xax, losses,'o-')
    pl.xlabel('iteration #')
    pl.tight_layout()
    pl.savefig(results_path + title + '.png', transparent = True)



      