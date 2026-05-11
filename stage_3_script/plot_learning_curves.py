import pickle
import matplotlib.pyplot as plt
import os

RESULT_DIR = '../../result/stage_3_result/'
PLOT_DIR = '../../result/stage_3_result/'

os.makedirs(PLOT_DIR, exist_ok=True)

datasets = ['MNIST', 'ORL', 'CIFAR']

# Individual plots, one per dataset
for name in datasets:

    file_path = os.path.join(RESULT_DIR, f'{name}_prediction_result')

    with open(file_path, 'rb') as f:
        result = pickle.load(f)

    loss_list = result['loss_list']
    acc_list = result['acc_list']
    epochs = range(1, len(loss_list) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, loss_list, marker='o', markersize=3)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Training Loss')
    ax1.set_title(f'{name}: Training Loss')
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, acc_list, marker='o', markersize=3, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Training Accuracy')
    ax2.set_title(f'{name}: Training Accuracy')
    ax2.set_ylim(0, 1.0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(PLOT_DIR, f'{name}_learning_curve.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()

    print(f'Saved: {out_path}')

# Combined loss plot showing all three on one figure for comparison
fig, ax = plt.subplots(figsize=(8, 5))

for name in datasets:
    file_path = os.path.join(RESULT_DIR, f'{name}_prediction_result')
    with open(file_path, 'rb') as f:
        result = pickle.load(f)

    loss_list = result['loss_list']
    epochs = range(1, len(loss_list) + 1)
    ax.plot(epochs, loss_list, marker='o', markersize=3, label=name)

ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('Training Loss Across Datasets')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = os.path.join(PLOT_DIR, 'all_datasets_loss_comparison.png')
plt.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close()
print(f'Saved: {out_path}')

print('\nAll plots saved to:', PLOT_DIR)