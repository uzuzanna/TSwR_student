from copy import copy
import numpy as np


class ESO:
    def __init__(self, A, B, W, L, state, Tp):
        self.A = A
        self.B = B
        self.W = W
        self.L = L
        self.state = np.pad(np.array(state), (0, A.shape[0] - len(state)), 'constant')
        self.Tp = Tp
        self.states = []
        self.u = 0.0

    def set_B(self, B):
        self.B = B

    # adrc
    # def update(self, q, u):
    #     self.states.append(copy(self.state))
    #
    #     z_hat = float(self.W @ self.state)
    #     error = float(q) - z_hat
    #     u = float(u)
    #
    #     state_dot = self.A @ self.state + self.B.flatten() * u + self.L.flatten() * error
    #
    #     # Całkowanie
    #     self.state = self.state + state_dot * self.Tp
    #
    #     self.u = u

    #adrcflc
    def update(self, q, u):
        self.states.append(copy(self.state))

        z_hat = self.W @ self.state
        error = q - z_hat

        u_arr = np.array(u).flatten()

        state_dot = self.A @ self.state + self.B @ u_arr + self.L @ error

        self.state = self.state + state_dot * self.Tp
        self.u = u

    def get_state(self):
        return self.state