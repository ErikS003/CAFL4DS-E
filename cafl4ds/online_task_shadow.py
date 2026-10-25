"""Causal task-outcome observer; never returns a training decision."""
import numpy as np


class TaskShadow:
    def __init__(self, prefix=128, state=None):
        self.prefix = prefix
        self.gamma = 2 ** (-1 / 256)
        self.count = 0
        self.buffer_x, self.buffer_y = [], []
        self.ready = False
        self.residual_abs = 0.0
        if state:
            self.__dict__.update(state)
            for key in ('center', 'scale', 'regularizer', 'a', 'b', 'fixed'):
                if key in state:
                    setattr(self, key, np.array(state[key]))

    def state(self):
        return {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in self.__dict__.items()}

    def transform(self, features):
        return np.r_[1., np.clip((features-self.center)/self.scale, -5, 5)]

    def predict(self, features):
        if not self.ready:
            return np.zeros(4)
        x = self.transform(features)
        return np.array([0., self.mean, x @ self.fixed * self.target_scale,
                         x @ np.linalg.solve(self.a, self.b) * self.target_scale])

    def _update(self, x, y):
        self.a = self.gamma*self.a + (1-self.gamma)*self.regularizer + np.outer(x, x)
        self.b = self.gamma*self.b + x*y/self.target_scale

    def learn(self, features, outcome, prediction):
        if not self.ready:
            self.buffer_x.append(np.asarray(features).tolist())
            self.buffer_y.append(float(outcome))
            self.count += 1
            if self.count == self.prefix:
                xx, yy = np.array(self.buffer_x), np.array(self.buffer_y)
                self.center, self.scale = xx.mean(0), xx.std(0)
                self.scale[self.scale < 1e-10] = 1
                self.target_scale = float(yy.std())
                if self.target_scale < 1e-12:
                    self.target_scale = 1.
                self.regularizer = np.diag([.1]+[10.]*xx.shape[1])
                self.a, self.b = self.regularizer.copy(), np.zeros(xx.shape[1]+1)
                for j, (features, y) in enumerate(zip(xx, yy)):
                    self._update(self.transform(features), y)
                    self.mean = float(y) if j == 0 else self.gamma*self.mean+(1-self.gamma)*y
                self.fixed = np.linalg.solve(self.a, self.b)
                self.residual_abs = float(np.mean(abs(yy-self.mean)))
                self.ready = True
                self.buffer_x, self.buffer_y = [], []
            return
        self._update(self.transform(features), outcome)
        self.mean = self.gamma*self.mean+(1-self.gamma)*outcome
        self.residual_abs = self.gamma*self.residual_abs+(1-self.gamma)*abs(outcome-prediction)
        self.count += 1
