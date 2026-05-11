import torch
import torch.nn as nn
import numpy as np
from code.stage_3_code.Evaluate_Accuracy import Evaluate_Accuracy
from code.base_class.method import method


class Method_CNN_ORL(method, nn.Module):

    def __init__(self, mName=None, mDescription=None):

        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)

        # Hyperparameters
        # ORL is tiny (400 images, 40 classes, ~10 per class) so we keep
        # the model small, use heavy dropout, and a small batch size
        self.learning_rate = 1e-3
        self.max_epoch = 80
        self.batch_size = 16

        self.loss_list = []
        self.acc_list = []

        # Architecture: input (N, 1, 112, 92), 40 classes
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),    # (N, 16, 112, 92)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 16, 56, 46)

            nn.Conv2d(16, 32, kernel_size=3, padding=1),   # (N, 32, 56, 46)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 32, 28, 23)

            nn.Conv2d(32, 64, kernel_size=3, padding=1),   # (N, 64, 28, 23)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # (N, 64, 14, 11)
        )

        # 64 * 14 * 11 = 9856
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 14 * 11, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 40),
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    def train_model(self, X, y):

        X_t = torch.FloatTensor(np.array(X))
        y_t = torch.LongTensor(np.array(y))

        self.train()  # PyTorch: set to training mode

        accuracy_evaluator = Evaluate_Accuracy('train_acc', '')

        for epoch in range(self.max_epoch):

            perm = torch.randperm(X_t.size(0))
            epoch_loss = 0
            batches = 0

            for i in range(0, X_t.size(0), self.batch_size):
                idx = perm[i:i + self.batch_size]
                X_b, y_b = X_t[idx], y_t[idx]

                self.optimizer.zero_grad()
                outputs = self.forward(X_b)
                loss = self.loss_fn(outputs, y_b)
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                batches += 1

            avg_loss = epoch_loss / batches
            self.loss_list.append(avg_loss)

            self.eval()  # PyTorch: set to eval mode for accuracy check
            with torch.no_grad():
                preds = []
                for i in range(0, X_t.size(0), 256):
                    preds.append(self.forward(X_t[i:i + 256]).max(1)[1])
                train_pred = torch.cat(preds)
            self.train()  # back to training mode

            accuracy_evaluator.data = {'true_y': y_t, 'pred_y': train_pred}
            metrics = accuracy_evaluator.evaluate()
            acc = metrics['accuracy']
            self.acc_list.append(acc)

            print(f"Epoch {epoch + 1}/{self.max_epoch} | Loss: {avg_loss:.4f} | Acc: {acc:.4f}")

    def test(self, X):

        self.eval()
        X_t = torch.FloatTensor(np.array(X))
        preds = []
        with torch.no_grad():
            for i in range(0, X_t.size(0), 256):
                preds.append(self.forward(X_t[i:i + 256]).max(1)[1])
        return torch.cat(preds).numpy()

    def run(self):
        print("method running...")
        print("--start training...")
        self.train_model(self.data['train']['X'], self.data['train']['y'])

        print("--start testing...")
        pred_y = self.test(self.data['test']['X'])

        return {
            'pred_y': pred_y,
            'true_y': self.data['test']['y'],
            'loss_list': self.loss_list,
            'acc_list': self.acc_list,
        }