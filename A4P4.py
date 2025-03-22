# Main program to dynamically choose between CPU/GPU/manual/high-level versions

def choose_version():
    print("Welcome to A4P4!")
    print("Would you like to use the high-level version (optimized and faster) or the manual version (slower but fully custom)?")
    choice = input("Enter 'manual' or 'highlevel': ").strip().lower()
    while choice not in ['manual', 'highlevel']:
        print("Invalid choice. Please enter 'manual' or 'highlevel'.")
        choice = input("Enter 'manual' or 'highlevel': ").strip().lower()
    return choice

def check_cupy_and_gpu():
    print("\nYou selected the manual version.")
    print("Checking system compatibility for GPU acceleration...")
    try:
        import cupy as cp
        is_cupy_available = cp.__version__
        has_gpu = cp.cuda.runtime.getDeviceCount() > 0
        if has_gpu:
            print("CuPy is installed, and your machine has a GPU.")
            use_gpu = input("Would you like to use the GPU version or the CPU version? Default is CPU. (Enter 'gpu' or 'cpu'): ").strip().lower()
            if use_gpu == 'gpu':
                print("Using the GPU manual implementation.")
                return 'manual_cnn(gpu)'
            else:
                print("Using the CPU manual implementation.")
                return 'manual_cnn(cpu)'
        else:
            print("CuPy is installed, but no GPU is available.")
            print("Defaulting to the CPU manual implementation.")
            return 'manual_cnn(cpu)'
    except ImportError:
        print("CuPy is not installed. Using the CPU manual implementation.")
        return 'manual_cnn(cpu)'

def plot_validation_loss(histories, labels):
    import matplotlib.pyplot as plt
    for history, label in zip(histories, labels):
        plt.plot(history, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.legend()
    plt.show()

def main():
    choice = choose_version()
    if choice == 'manual':
        version = check_cupy_and_gpu()
        if version == 'manual_cnn(gpu)':
            import manual_cnn_gpu as cnn
        else:
            import manual_cnn_cpu as cnn
    else:
        import highlevel_cnn as cnn

    # Load and preprocess data
    x_train, y_train, x_val, y_val, x_test, y_test = cnn.load_and_preprocess()
    cnn.visualize_samples(x_train, y_train)

    # Train models with different feature extractors and track histories
    histories = []
    labels = ['Original', 'Gaussian Noise', 'Sharpened']

    # Original images
    print("Training with original images...")
    _, acc_original = cnn.train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test)
    histories.append([])  # Placeholder for validation loss history
    print(f"Test Accuracy (Original): {acc_original:.4f}")

    # Images with Gaussian noise
    print("Training with images with Gaussian noise...")
    _, acc_noisy = cnn.train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, cnn.add_gaussian_noise)
    histories.append([])  # Placeholder for validation loss history
    print(f"Test Accuracy (Gaussian Noise): {acc_noisy:.4f}")

    # Sharpened images
    print("Training with sharpened images...")
    _, acc_sharpened = cnn.train_and_evaluate(x_train, y_train, x_val, y_val, x_test, y_test, cnn.sharpen_images)
    histories.append([])  # Placeholder for validation loss history
    print(f"Test Accuracy (Sharpened): {acc_sharpened:.4f}")

    # Plot validation loss
    print("Plotting validation loss...")
    plot_validation_loss(histories, labels)

    # Print final accuracies
    print("Final Test Accuracies:")
    print(f"Original: {acc_original:.4f}")
    print(f"Gaussian Noise: {acc_noisy:.4f}")
    print(f"Sharpened: {acc_sharpened:.4f}")

if __name__ == "__main__":
    main()
