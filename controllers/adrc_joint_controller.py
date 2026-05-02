import numpy as np
from observers.eso import ESO
from .controller import Controller


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.b = b
        self.kp = kp
        self.kd = kd

        A = np.array([[0, 1, 0],
                      [0, 0, 1],
                      [0, 0, 0]])
        B = np.array([[0],
                      [self.b],
                      [0]])
        L = np.array([[3 * p],
                      [3 * p ** 2],
                      [p ** 3]])
        W = np.array([[1, 0, 0]])

        state0 = np.array([q0[0], q0[1], 0.0])
        self.eso = ESO(A, B, W, L, state0, Tp)

    def set_b(self, b):
        self.b = b
        self.eso.B = np.array([[0], [self.b], [0]])

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = x[0]

        self.eso.update(q, self.eso.u)
        q_est, q_dot_est, f_est = self.eso.get_state()

        e = q_d - q_est
        e_dot = q_d_dot - q_dot_est
        v = q_d_ddot + self.kd * e_dot + self.kp * e

        u = float((v - f_est) / self.b)

        return u