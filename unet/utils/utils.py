import matplotlib.pyplot as plt


def plot_img_and_mask(img, mask, out_filename, classes_1=False):
    
    if not classes_1:
        classes = mask.max() + 1
    else:
        classes = 2
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.savefig(f'{out_filename}.png')
  
    
    
def plot_img_and_mask_1class(img, mask, out_filename):
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.savefig(f'{out_filename}.png')