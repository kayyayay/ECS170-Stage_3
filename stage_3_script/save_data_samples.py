import pickle
import matplotlib.pyplot as plt
import os

# save sample images from each dataset for inclusion in the report
OUT_DIR = '../../result/stage_3_result/data_samples/'
os.makedirs(OUT_DIR, exist_ok=True)

datasets = [
    ('MNIST', 'gray'),
    ('ORL',   'gray'),
    ('CIFAR', None),
]

for dataset_name, cmap in datasets:
    file_path = f'../../data/stage_3_data/{dataset_name}'

    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    fig, axes = plt.subplots(1, 6, figsize=(12, 2.5))
    for i, instance in enumerate(data['train'][:6]):
        img = instance['image']
        label = instance['label']
        axes[i].imshow(img, cmap=cmap)
        axes[i].set_title(f'Label: {label}', fontsize=10)
        axes[i].axis('off')

    plt.suptitle(f'{dataset_name} sample training images', fontsize=12)
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, f'{dataset_name}_samples.png')
    plt.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f'Saved {out_path}')

print('\nAll sample images saved to:', OUT_DIR)