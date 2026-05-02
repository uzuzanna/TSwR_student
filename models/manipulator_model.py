import numpy as np

class ManiuplatorModel:
    def __init__(self, Tp):
        self.Tp = Tp
        self.l1 = 0.5
        self.m1 = 3.0
        self.I_1 = 0.06370

        self.l2 = 0.4
        self.m2 = 2.4
        self.I_2 = 0.03296

        self.m3 = 0.1
        self.I_3 = 0.00010

    def M(self, x):
        """
        Calculates the mass matrix
        """
        q1, q2, q1_dot, q2_dot = x
        d1 = self.l1 / 2
        d2 = self.l2 / 2
        alpha = self.I_1 + self.I_2 + self.m1 * d1 ** 2 + self.m2 * (self.l1 ** 2 + d2 ** 2) + \
                self.I_3 + self.m3 * (self.l1 ** 2 + self.l2 ** 2)
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        delta = self.I_2 + self.m2 * d2 ** 2 + self.I_3 + self.m3 * self.l2 ** 2
        m_11 = alpha + 2 * beta * np.cos(q2)
        m_12 = delta + beta * np.cos(q2)
        m_21 = m_12
        m_22 = delta
        return np.array([[m_11, m_12], [m_21, m_22]])

    def C(self, x):
        """
        Calculates the Coriolis and centrifugal forces matrix
        """
        q1, q2, q1_dot, q2_dot = x
        d2 = self.l2 / 2
        beta = self.m2 * self.l1 * d2 + self.m3 * self.l1 * self.l2
        c_11 = -beta * np.sin(q2) * q2_dot
        c_12 = -beta * np.sin(q2) * (q1_dot + q2_dot)
        c_21 = beta * np.sin(q2) * q1_dot
        c_22 = 0
        return np.array([[c_11, c_12], [c_21, c_22]])