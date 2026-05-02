import matplotlib.pyplot as plt
import numpy as np

from controllers.feedback_linearization_controller import FeedbackLinearizationController
from trajectory_generators.constant_torque import ConstantTorque
from trajectory_generators.sinusonidal import Sinusoidal
from trajectory_generators.poly3 import Poly3
from utils.simulation import simulate

Tp = 0.01
start = 0
end = 3

controller = FeedbackLinearizationController(Tp)

# Trajektoria
traj_gen = Sinusoidal(np.array([0., 1.]), np.array([2., 2.]), np.array([0., 0.]))

Q, Q_d, u, T = simulate("PYBULLET", traj_gen, controller, Tp, end)

# Rysowanie wykresow
plt.subplot(221)
plt.plot(T, Q[:, 0], 'r')
plt.plot(T, Q_d[:, 0], 'b')
plt.title("Pozycja q1")

plt.subplot(222)
plt.plot(T, Q[:, 1], 'r')
plt.plot(T, Q_d[:, 1], 'b')
plt.title("Pozycja q2")

plt.subplot(223)
plt.plot(T, u[:, 0], 'r')
plt.plot(T, u[:, 1], 'b')
plt.title("Sterowanie u")

plt.tight_layout()
plt.show()