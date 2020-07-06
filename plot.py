
import matplotlib.pyplot as plt
import numpy as np

def plot_keras_history(history, epochs, metric, vali_freq = None):
    acc = history.history[metric]
    val_acc = history.history['val_' + metric]

    epochs_range = range(epochs)
    if vali_freq != None:
        temp = []
        j = 0
        for i in epochs_range:
            k = i + 1
            if k % vali_freq == 0:
                temp.append(val_acc[j])
                j += 1
            else:
                if len(temp) == 0:
                    temp.append(val_acc[0])
                else:   
                    temp.append(temp[i - 1])
                    
        val_acc = np.array(temp)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training ' + metric)
    plt.plot(epochs_range, val_acc, label='Validation ' + metric)
    plt.legend(loc='lower right')
    plt.title('Training and Validation ' + metric)
    
    acc_string = 'acc'
    if acc_string in metric:
        loss=history.history['loss']
        val_loss=history.history['val_loss']
        
        if vali_freq != None:
            temp = []
            j = 0
            for i in epochs_range:
                k = i + 1
                if k % vali_freq == 0:
                    temp.append(val_loss[j])
                    j += 1
                else:
                    if len(temp) == 0:
                        temp.append(val_loss[0])
                    else:   
                        temp.append(temp[i - 1])
                    

            val_loss = np.array(temp)

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
    plt.show()
    
    
def show_images(images, cols = 1, titles = None):
    assert((titles is None)or (len(images) == len(titles)))
    n_images = len(images)
    if titles is None: titles = ['Image (%d)' % i for i in range(1,n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()