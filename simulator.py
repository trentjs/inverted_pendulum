''' 
=== Inverted Pendulum Simulator ===

This module does blar...
'''

__created__= "Monday 17th December, 2020"
__author__ = "Trent Jansen-Sturgeon"
__email__ = "trentjansensturgeon@gmail.com"
__version__ = "1.0"
__scriptName__ = "inverted_pendulum_simulator.py"
__status__ = "Construction"

# IO modules
import argparse

# Science modules
import numpy as np
from scipy.integrate import solve_ivp
from simple_pid import PID

# Plotting modules
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('seaborn-pastel')

# Constants
g = 9.81 #m/s^2

def dynamics(dt, state, f):
    # Resource to check my answer:
    # https://metr4202.uqcloud.net/tpl/t8-Week13-pendulum.pdf

    # Decompose the state
    x, x_dot, thi, thi_dot, m, M, l = state
    c, s = np.cos(thi), np.sin(thi)
    denom = m*s**2 + M # Shorthand

    # f = 0 # Include the force on the cart later...
    
    # Determine the second order terms
    x_ddot = (m*s*(l*thi_dot**2 - g*c) + f) / denom
    thi_ddot = ((m+M)*g*s - m*l*s*c*thi_dot**2 - f*c) / (l*denom)
    state_dot = np.array([x_dot, x_ddot, thi_dot, thi_ddot, 0, 0, 0])

    return state_dot


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Inverted Pendulum Simulator")
    parser.add_argument("-x0", "--x_initial", type=float, default=0,
        help="The initial position of the cart [meters].")
    parser.add_argument("-x_dot0", "--x_dot_initial", type=float, default=0,
        help="The initial velocity of the cart [meters/second].")
    parser.add_argument("-thi0", "--thi_initial", type=float, default=0,
        help="The initial angle offset of the pendulum [degrees].")
    parser.add_argument("-thi_dot0", "--thi_dot_initial", type=float, default=100,
        help="The initial angular velocity of the cart [degrees/second].")
    parser.add_argument("-m", "--pendulum_mass", type=float, default=1,
        help="The mass of the pendulum [kg].")
    parser.add_argument("-M", "--cart_mass", type=float, default=2,
        help="The mass of the cart [kg].")
    args = parser.parse_args()

    # Initialise the state
    x0, x_dot0 = args.x_initial, args.x_dot_initial
    thi0 = np.deg2rad(args.thi_initial)
    thi_dot0 = np.deg2rad(args.thi_dot_initial)
    m, M = args.pendulum_mass, args.cart_mass
    state = np.array([x0, x_dot0, thi0, thi_dot0, m, M, 1])

    # Define the desired state
    x_desired, thi_desired = 0, 0
    k = 1 / np.deg2rad(1)
    
    # Setup the state array to fill
    tot_time, dt = 20, 0.1
    n_times = int(tot_time / dt)
    times = np.linspace(0,tot_time,n_times)
    states = np.tile(state,(n_times,1))
    force = np.zeros(n_times)
    
    pid = PID(1, 0.1, 0.05, setpoint=0)

    for i in range(1,n_times):

        # Determine the error using cost fn = dx + k*dthi
        error = (states[i-1,0] - x_desired) \
                + k * (states[i-1,2] - thi_desired + np.pi)%(2*np.pi) - np.pi

        # Conver the error to force (PID controller)
        force[i] = pid(error)

        # Simulate the response by solving the initial value problem
        states[i] = solve_ivp(dynamics, (0,dt), states[i-1],
                              args=(force[i],), method='DOP853').y[:,-1]

    # Create the simulation video
    x_range, y_range = 5, 1.2
    fig = plt.figure(figsize=(12,12*y_range/x_range))
    ax = plt.axes(xlim=(-x_range,x_range), ylim=(-y_range,y_range))
    cart = plt.Rectangle([x0-0.1,-0.05], 0.2, 0.1, color='g', zorder=10)
    link, = ax.plot([x0,x0 + np.sin(thi0)], [0,np.cos(thi0)], c='b', zorder=20)
    pendulum = plt.Circle((x0 + np.sin(thi0), np.cos(thi0)), 0.05, color='r', zorder=30)
    ax.add_artist(cart); ax.add_artist(pendulum); ax.set_aspect('equal'); ax.grid()

    def animate(i):
        x, thi, l = states[i,[0,2,6]]
        cart.set_xy([x-0.1,-0.05])
        link.set_data([x,x + l*np.sin(thi)], [0,l*np.cos(thi)])
        pendulum.set_center([x + l*np.sin(thi), l*np.cos(thi)])
        return cart, link, pendulum

    anim = FuncAnimation(fig, animate, frames=n_times)
    anim.save('inverted_pendulum.mp4', writer='ffmpeg', fps=1/dt)

    # Plot x and thi over time
    plt.figure()
    plt.subplot(3,1,1); plt.grid()
    plt.plot(times, states[:,0],'.-')
    plt.xlabel('Time [sec]'); plt.ylabel('X [m]')
    plt.subplot(3,1,2); plt.grid()
    plt.plot(times, np.rad2deg(states[:,2]),'.-')
    plt.xlabel('Time [sec]'); plt.ylabel('Thi [deg]')
    plt.subplot(3,1,3); plt.grid()
    plt.plot(times, force,'.-')
    plt.xlabel('Time [sec]'); plt.ylabel('Force [N]')
    plt.show()

    # Plot the phase plots
    plt.figure()
    plt.subplot(1,2,1); plt.grid()
    plt.plot(states[:,0], states[:,1],'.-')
    plt.xlabel('x'); plt.ylabel('x_dot')
    plt.subplot(1,2,2); plt.grid()
    plt.plot(np.rad2deg(states[:,2]), np.rad2deg(states[:,3]),'.-')
    plt.xlabel('thi'); plt.ylabel('thi_dot')
    plt.show()

