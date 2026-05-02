import numpy as np
from models.manipulator_model import ManiuplatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManiuplatorModel(Tp)
        self.Kp = np.array([[200, 0], [0, 200]])
        self.Kd = np.array([[80, 0], [0, 80]])

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Feedback linearization using self.model
        """
        # Wyciągamy aktualne pozycje (q) i prędkości (q_dot) ze stanu robota (x)
        q = np.array([x[0], x[1]])
        q_dot = np.array([x[2], x[3]])

        q_r = np.array(q_r)
        q_r_dot = np.array(q_r_dot)
        q_r_ddot = np.array(q_r_ddot)

        # Liczymy uchyb pozycji i prędkości
        e = q_r - q
        e_dot = q_r_dot - q_dot

        # Sygnał zewnętrznego kontrolera PD
        v = q_r_ddot + self.Kd @ e_dot + self.Kp @ e

        # Wywołujemy macierze z modelu, podając pełny wektor stanu x
        M = self.model.M(x)
        C = self.model.C(x)

        # Równanie FLC kasujące nieliniowości
        tau = M @ v + C @ q_dot

        return tau