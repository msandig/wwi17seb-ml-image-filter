
import matplotlib.pyplot as plt

def plot_keras_history(history, epochs, metric):
    acc = history.history[metric]
    val_acc = history.history['val_' + metric]



    epochs_range = range(epochs)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training ' + metric)
    plt.plot(epochs_range, val_acc, label='Validation ' + metric)
    plt.legend(loc='lower right')
    plt.title('Training and Validation ' + metrics)
    
    if 'acc' in metric:
        loss=history.history['loss']
        val_loss=history.history['val_loss']

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
    plt.show()