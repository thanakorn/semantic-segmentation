import matplotlib.pyplot as plt

DEFAULT_FIG_SIZE = (4, 4)

def plot_losses(training_losses, validation_losses, fig_size=DEFAULT_FIG_SIZE):
    num_epochs = len(training_losses)
    plt.figure(figsize=fig_size)
    plt.title('Losses')
    plt.plot(list(range(1, num_epochs + 1)), training_losses, label='Train')
    plt.plot(list(range(1, num_epochs + 1)), validation_losses, label='Validation')
    plt.legend()