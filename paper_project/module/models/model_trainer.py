import os
import pickle
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from mealpy.swarm_based import SSA

from module.models.elm import elm_gpu, compute_mmd


class ModelTrainer:
    def __init__(self, args):
        self.args = args

    def _load_or_optimize_parameters(self, name, train_sc_x, train_sc_y,
                                      return_history=False):
        if self.args.is_opt == 1:
            sol, history = self._optimization(name, train_sc_x, train_sc_y)
            return (sol, history) if return_history else sol
        elif self.args.is_opt == 0:
            with open(os.path.join(self.args.args_path, f'{name}.pkl'), 'rb') as f:
                sol = pickle.load(f)
            return (sol, None) if return_history else sol
        return (self.args.p, None) if return_history else self.args.p

    def _model_set(self, name, train_sc_x, train_sc_y, shape):
        return self._build_elm_model(name, train_sc_x, train_sc_y, shape)

    def _build_elm_model(self, name, train_sc_x, train_sc_y, shape):
        model = elm_gpu(
            hidden_units=int(self.args.p['elm_filter']),
            activation_function='relu',
            x=train_sc_x,
            y=train_sc_y,
            C2=self.args.p['C2'],
            elm_type=self.args.elm_type,
            mmd_weight=self.args.mmd_weight,
            rmse_weight=self.args.rmse_weight,
        )
        model.fit(algorithm='solution2')
        joblib.dump(model, self.args.models_path + name + '.pkl')
        return model

    def _optimization(self, name, train_sc_x, train_sc_y):
        X_train, X_val, y_train, y_val = train_test_split(
            train_sc_x, train_sc_y, test_size=self.args.val_len, shuffle=False
        )

        LB = self.args.ssa_bounds['LB']
        UB = self.args.ssa_bounds['UB']

        def decode(solution):
            return {
                'elm_filter': int(solution[0]),
                'C2': solution[1],
            }

        def fitness_function(solution):
            for key, value in decode(solution).items():
                self.args.p[key] = value
            model = self._model_set(name, X_train, y_train, X_train.shape)
            y_pred = model.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mmd = compute_mmd(y_val, y_pred)
            if self.args.elm_type == 'reg':
                return self.args.rmse_weight * rmse
            else:
                return self.args.rmse_weight * rmse + self.args.mmd_weight * mmd

        problem = {
            "fit_func": fitness_function,
            "lb": LB,
            "ub": UB,
            "minmax": "min",
            "log_to": None,
            "save_population": False,
        }

        optimizer = SSA.BaseSSA(epoch=100, pop_size=20)
        optimizer.solve(problem)

        sol = decode(optimizer.solution[0])

        fitness_history = list(optimizer.history.list_global_best_fit)

        # Extract per-epoch best decoded parameters from history
        params_history = []
        for agent in optimizer.history.list_global_best:
            # mealpy agent: [position_array, [fitness_value]]
            position = agent[0] if isinstance(agent, (list, tuple)) else agent.solution
            params_history.append(decode(position))

        history_data = {
            'fitness': fitness_history,
            'params': params_history,
        }

        if self.args.is_save:
            with open(os.path.join(self.args.args_path, f'{name}.pkl'), 'wb') as f:
                pickle.dump(sol, f)
        print('优化结束!', sol)
        return sol, history_data
