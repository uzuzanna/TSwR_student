import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        self.Tp = Tp
        self.models = [ManiuplatorModel(Tp), ManiuplatorModel(Tp), ManiuplatorModel(Tp)]

        params = [(0.1, 0.05), (0.01, 0.01), (1.0, 0.3)]
        for i, (m3, r3) in enumerate(params):
            self.models[i].m3 = m3
            self.models[i].r3 = r3
            self.models[i].I_3 = 2. / 5 * m3 * r3 ** 2

        self.i = 0

        self.Kp = np.array([[100, 0], [0, 100]])
        self.Kd = np.array([[20, 0], [0, 20]])

        self.u_prev = np.zeros(2)
        self.q_dot_prev = np.zeros(2)

    def choose_model(self, x):
        q_dot = x[2:]

        q_ddot_real = (q_dot - self.q_dot_prev) / self.Tp

        errors = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)

            q_ddot_pred = np.linalg.inv(M) @ (self.u_prev - C @ self.q_dot_prev)

            errors.append(np.sum((q_ddot_real - q_ddot_pred) ** 2))

        self.i = np.argmin(errors)
        self.q_dot_prev = q_dot

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)

        q = x[:2]
        q_dot = x[2:]

        e = q_r - q
        e_dot = q_r_dot - q_dot

        v = q_r_ddot + self.Kd @ e_dot + self.Kp @ e

        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)

        u = M @ v + C @ q_dot

        self.u_prev = u

        return u