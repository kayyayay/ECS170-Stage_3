'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting


class Setting_CNN(setting):

    def load_run_save_evaluate(self):

        # load dataset
        loaded_data = self.dataset.load()

        # run the CNN method
        self.method.data = {
            'train': {'X': loaded_data['X_train'], 'y': loaded_data['y_train']},
            'test':  {'X': loaded_data['X_test'],  'y': loaded_data['y_test']},
        }
        learned_result = self.method.run()

        # evaluate on test predictions
        self.evaluate.data = {
            'true_y': learned_result['true_y'],
            'pred_y': learned_result['pred_y'],
        }
        metrics = self.evaluate.evaluate()

        # save predictions, learning curves, and metrics together
        self.result.data = {
            'pred_y': learned_result['pred_y'],
            'true_y': learned_result['true_y'],
            'loss_list': learned_result['loss_list'],
            'acc_list': learned_result['acc_list'],
            'metrics': metrics,
        }
        self.result.save()

        return metrics