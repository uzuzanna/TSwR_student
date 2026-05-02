import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManiuplatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd

        p1 = p[0]
        p2 = p[1]

        self.L = np.array([[3 * p1, 0],
                           [0, 3 * p2],
                           [3 * p1 ** 2, 0],
                           [0, 3 * p2 ** 2],
                           [p1 ** 3, 0],
                           [0, p2 ** 3]])

        W = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0]])

        A = np.zeros((6, 6))
        A[0, 2] = 1
        A[1, 3] = 1
        A[2, 4] = 1
        A[3, 5] = 1

        B = np.zeros((6, 2))

        state = np.array([q0[0], q0[1], q0[2], q0[3], 0.0, 0.0])

        self.eso = ESO(A, B, W, self.L, state, Tp)
        self.u = np.zeros(2)  # Poczatkowe u
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):

        x = [q[0], q[1], q_dot[0], q_dot[1]]
        M = self.model.M(x)
        M_inv = np.linalg.inv(M)

        B = np.zeros((6, 2))
        B[2:4, 0:2] = M_inv
        self.eso.set_B(B)

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        q = np.array([x[0], x[1]])
        q_dot = np.array([x[2], x[3]])

        self.update_params(q, q_dot)
        self.eso.update(q, self.u)

        state_est = self.eso.get_state()
        q_est = np.array([state_est[0], state_est[1]])
        q_dot_est = np.array([state_est[2], state_est[3]])
        f_est = np.array([state_est[4], state_est[5]])

        e = q_d - q_est
        e_dot = q_d_dot - q_dot_est

        v = q_d_ddot + self.Kd @ e_dot + self.Kp @ e

        M = self.model.M(x)
        C = self.model.C(x)

        u = M @ (v - f_est) + C @ q_dot
        self.u = u
        return u