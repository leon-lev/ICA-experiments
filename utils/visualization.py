import math

import matplotlib.pyplot as plt
import numpy as np


def plot_examples(dataset, n_examples=16):
    """
    Visualize a set of examples of images from the dataset.
    """
    n_rows = math.ceil(math.sqrt(n_examples))
    n_cols = math.ceil(n_examples / n_rows)
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(15, 15))

    for i, axx in enumerate(ax):
        if i >= n_examples:
            axx.axis('off')
            continue
        ind = np.random.randint(len(dataset))  
        img, cond = dataset[ind]
        axx.imshow(np.array(img.permute(1, 2, 0)))
        axx.set_title(f'Img #{ind}, Label: {cond}')
        axx.axis('off')
  
    plt.show()
    return fig, ax


def visualize_model_outputs(dataset, encoder, decoder,
                            labels, device, n_examples,
                            save_path=''):
    """
    Visualize model inputs and outputs.
    """

    l = len(labels)
    fig, ax = plt.subplots(nrows=n_examples,
                         ncols=l, 
                         figsize=(5 * n_examples, 5 * (l + 1)))
    for i in range(n_examples):
        ind = np.random.randint(len(dataset))  
        img, cond = dataset[ind]
        ax[i][0].imshow(np.array(img.permute(1, 2, 0)))
        ax[i][0].set_title(f'Img #{ind}, Label: {cond}')
        ax[i][0].axis('off')

        x, cond = img.unsqueeze(0).to(device), cond.to(device)
        recon, code, _ = encoder(x, cond)

        for j, cond_out in zip(range(1, l+1), labels):
            recon = decoder(code, cond_out)
            ax[i][j].imshow(np.array(recon.permute(1, 2, 0)))
            title = f'Reconstucted with label: {cond_out}'
            if cond_out == cond:
                title += ' (original)'
            ax[i][j].set_title(title)
            ax[i][j].axis('off')
    
    plt.show()
    if save_path:
        fig.savefig(save_path)
    return fig, ax