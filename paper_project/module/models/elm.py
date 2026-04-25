import torch
import numpy as np
from scipy.linalg import pinv, inv


def compute_mmd(x, y):
    if isinstance(x, torch.Tensor):
        x = x.cpu().detach().numpy()
    if isinstance(y, torch.Tensor):
        y = y.cpu().detach().numpy()

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)

    rx = np.diag(xx)
    ry = np.diag(yy)

    dxx = rx[:, np.newaxis] + rx[np.newaxis, :] - 2 * xx
    dyy = ry[:, np.newaxis] + ry[np.newaxis, :] - 2 * yy
    dxy = rx[:, np.newaxis] + ry[np.newaxis, :] - 2 * xy

    distances = np.sqrt(dxy[dxy > 0])
    sigma_med = np.median(distances) if len(distances) > 0 else 1.0

    bandwidth_multipliers = [0.1, 0.5, 1.0, 2.0, 10.0]
    sigmas = [b * sigma_med for b in bandwidth_multipliers]

    m = x.shape[0]
    n = y.shape[0]

    mmd_val = 0.0
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2 + 1e-8)
        k_xx = np.exp(-gamma * dxx)
        k_yy = np.exp(-gamma * dyy)
        k_xy = np.exp(-gamma * dxy)
        term_xx = np.sum(k_xx) / (m * m)
        term_yy = np.sum(k_yy) / (n * n)
        term_xy = 2 * np.sum(k_xy) / (m * n)
        mmd_val += (term_xx + term_yy - term_xy)

    mmd_val = mmd_val / len(sigmas)
    return np.sqrt(max(mmd_val, 0.0) + 1e-6)


class elm_gpu():
    '''
    Extreme Learning Machine (GPU Version)
    Parameters:
        hidden_units: int, number of hidden units
        activation_function: str, 'sigmoid', 'relu', 'sin', 'tanh' or 'leaky_relu'
        x: array/tensor, shape[samples, features]
        y: array/tensor, shape[samples, ]
        C2: float, regularization parameter
        elm_type: str, 'clf', 'reg', or 'custom'
        one_hot: bool, default True
        random_type: str, 'uniform' or 'normal', default 'normal'
        device: str, 'cuda' or 'cpu'
        mmd_weight: float, weight for MMD loss (used when elm_type='custom')
        rmse_weight: float, weight for RMSE loss (used when elm_type='custom')
    '''

    def __init__(self, hidden_units, activation_function, x, y, C2, elm_type,
                 one_hot=True, random_type='normal', device='cuda',
                 mmd_weight=0.1, rmse_weight=1.0):

        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        if not isinstance(y, torch.Tensor):
            y = torch.tensor(y, dtype=torch.long if elm_type == 'clf' else torch.float32)

        self.device = device
        self.hidden_units = hidden_units
        self.activation_function = activation_function
        self.random_type = random_type
        self.x = x.to(self.device)
        self.y = y.to(self.device)
        self.C = C2
        self.class_num = torch.unique(self.y).shape[0] if elm_type == 'clf' else 1
        self.beta = torch.zeros((hidden_units, self.class_num), device=self.device)
        self.elm_type = elm_type
        self.one_hot = one_hot
        self.mmd_weight = mmd_weight
        self.rmse_weight = rmse_weight

        if elm_type == 'clf' and self.one_hot:
            self.one_hot_label = torch.nn.functional.one_hot(self.y, num_classes=self.class_num).float()

        torch.manual_seed(8)
        if random_type == 'uniform':
            self.W = torch.empty((hidden_units, x.shape[1]), device=self.device).uniform_(-1, 1)
            self.b = torch.empty((hidden_units, 1), device=self.device).uniform_(-1, 1)
        elif random_type == 'normal':
            self.W = torch.randn((hidden_units, x.shape[1]), device=self.device) * 0.5
            self.b = torch.randn((hidden_units, 1), device=self.device) * 0.5

    def __input2hidden(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32, device=self.device)
        else:
            x = x.to(self.device)

        self.temH = torch.mm(self.W, x.t()) + self.b

        if self.activation_function == 'sigmoid':
            self.H = torch.sigmoid(self.temH)
        elif self.activation_function == 'relu':
            self.H = torch.relu(self.temH)
        elif self.activation_function == 'sin':
            self.H = torch.sin(self.temH)
        elif self.activation_function == 'tanh':
            self.H = torch.tanh(self.temH)
        elif self.activation_function == 'leaky_relu':
            self.H = torch.nn.functional.leaky_relu(self.temH, 0.1)
        return self.H.t()

    def __hidden2output(self, H):
        return torch.mm(H, self.beta)

    def fit(self, algorithm):
        with torch.no_grad():
            H = self.__input2hidden(self.x)
            if self.elm_type == 'clf' and self.one_hot:
                y_temp = self.one_hot_label
            else:
                y_temp = self.y

            I = torch.eye(self.hidden_units, device=self.device)
            eps = 1e-6

            if algorithm == 'no_re':
                self.beta = torch.linalg.lstsq(H, y_temp).solution
            elif algorithm == 'solution1':
                Ht_H = torch.mm(H.t(), H)
                regularized_matrix = I / self.C + Ht_H + eps * I
                try:
                    tmp = torch.inverse(regularized_matrix)
                except torch._C._LinAlgError:
                    tmp = torch.linalg.pinv(regularized_matrix)
                self.beta = torch.mm(torch.mm(tmp, H.t()), y_temp)
            elif algorithm == 'solution2':
                HHt = torch.mm(H.t(), H)
                regularized_matrix = I / self.C + HHt + eps * I
                try:
                    tmp = torch.inverse(regularized_matrix)
                except torch._C._LinAlgError:
                    tmp = torch.linalg.pinv(regularized_matrix)
                self.beta = torch.mm(torch.mm(H, tmp).t(), y_temp)

            output = self.__hidden2output(H)

            if self.elm_type == 'clf':
                output = torch.softmax(output, dim=1)
                self.train_score = (torch.argmax(output, dim=1) == self.y).float().mean().item()
            elif self.elm_type == 'reg':
                self.train_score = torch.sqrt(torch.mean((output - y_temp) ** 2)).item()
            elif self.elm_type == 'custom':
                rmse = torch.sqrt(torch.mean((output - y_temp) ** 2)).item()
                mmd = compute_mmd(output, y_temp)
                self.train_score = self.rmse_weight * rmse + self.mmd_weight * mmd

        return self.beta.cpu().numpy(), self.train_score

    def predict(self, x):
        with torch.no_grad():
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32, device=self.device)
            else:
                x = x.to(self.device)

            H = self.__input2hidden(x)
            output = self.__hidden2output(H)

            if self.elm_type == 'clf':
                y_pred = torch.argmax(output, dim=1)
            else:
                y_pred = output

        return y_pred.cpu().numpy()
