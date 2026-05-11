from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_MNIST_Variant import Method_CNN_MNIST_Variant
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_CNN import Setting_CNN
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

configs = [
    {'name': 'baseline'},
    {'name': 'kernel_5_valid',  'kernel_size': 5, 'padding': 0},
    {'name': 'kernel_5_same',   'kernel_size': 5, 'padding': 2},
    {'name': '1_block',         'num_blocks': 1},
    {'name': '3_blocks',        'num_blocks': 3},
    {'name': 'avg_pool',        'pool_type': 'avg'},
    {'name': 'narrower_fc',     'hidden_dim': 64},
    {'name': 'wider_fc',        'hidden_dim': 256},
    {'name': 'tanh',            'activation': 'tanh'},
    {'name': 'sigmoid',         'activation': 'sigmoid'},
]

results = []

for cfg in configs:
    name = cfg.pop('name')
    print(f"\n========== Config: {name} ==========")

    np.random.seed(2)
    torch.manual_seed(2)

    data_obj = Dataset_Loader('MNIST', '')
    data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
    data_obj.dataset_source_file_name = 'MNIST'
    data_obj.dataset_name = 'MNIST'

    method_obj = Method_CNN_MNIST_Variant('CNN', '', **cfg)

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_3_result/'
    result_obj.result_destination_file_name = f'MNIST_ablation_{name}'

    setting_obj = Setting_CNN('CNN setting', '')
    evaluate_obj = Evaluate_Accuracy('accuracy', '')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    metrics = setting_obj.load_run_save_evaluate()

    results.append((name, metrics))

print("\n========== Ablation Summary ==========")
print(f"{'config':<20} {'accuracy':>10} {'precision':>10} {'recall':>10} {'f1':>10}")
for name, m in results:
    print(f"{name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} {m['recall']:>10.4f} {m['f1']:>10.4f}")