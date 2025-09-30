import matplotlib.pyplot as plt


def plot_val_and_test(logs, labels, no_limit=False):
    # Determine the minimum number of epochs across all logs
    min_max_epochs = min(max(log.metrics['epoch'].unique()) + 1 for log in logs)
    epochs = range(1, min_max_epochs + 1)

    # Set up subplots
    fig, axes = plt.subplots(2, 1, figsize=(6, 5), sharex=True, sharey=True)

    # Plot Pitts30k-val
    for log, label in zip(logs, labels):
        if no_limit:
            epochs = log.metrics['epoch'].unique() + 1
            val_data = log.metrics['pitts30k_val/R1'].dropna()
        else:
            val_data = log.metrics['pitts30k_val/R1'].dropna()[:min_max_epochs]
        axes[0].plot(epochs, val_data,'-*', label=label)
    axes[0].set_title("Pitts30k-val dataset Results")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Recall@1")
    axes[0].grid(True)
    axes[0].legend()

    # Plot Pitts30k-test
    for log, label in zip(logs, labels):
        if no_limit:
            epochs = log.metrics['epoch'].unique() + 1
            test_data = log.metrics['pitts30k_test/R1'].dropna()
        else:
            test_data = log.metrics['pitts30k_test/R1'].dropna()[:min_max_epochs]
        axes[1].plot(epochs, test_data,'-*' , label=label)
    axes[1].set_title("Pitts30k-test dataset Results")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Recall@1")
    axes[1].grid(True)
    axes[1].legend()
    

    plt.tight_layout()
    plt.show()
