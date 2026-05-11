from code.stage_3_code.Dataset_Loader import Dataset_Loader
from code.stage_3_code.Method_CNN_ORL import Method_CNN_ORL
from code.stage_3_code.Result_Saver import Result_Saver
from code.stage_3_code.Setting_CNN import Setting_CNN
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch

# ---- reproducibility ----
np.random.seed(2)
torch.manual_seed(2)

# ---- objects ----
data_obj = Dataset_Loader('ORL', '')
data_obj.dataset_source_folder_path = '../../data/stage_3_data/'
data_obj.dataset_source_file_name = 'ORL'
data_obj.dataset_name = 'ORL'

method_obj = Method_CNN_ORL('CNN', '')

result_obj = Result_Saver('saver', '')
result_obj.result_destination_folder_path = '../../result/stage_3_result/'
result_obj.result_destination_file_name = 'ORL_prediction_result'

setting_obj = Setting_CNN('CNN setting', '')
evaluate_obj = Evaluate_Accuracy('accuracy', '')

# ---- run ----
setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
setting_obj.print_setup_summary()
metrics = setting_obj.load_run_save_evaluate()

print('================ ORL CNN Results ================')
print(f"Accuracy:  {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall:    {metrics['recall']:.4f}")
print(f"F1:        {metrics['f1']:.4f}")