# Copyright (c) 2024 Takumi Asada
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------------
# Read urdf_to_dh parameters from csv file.
# --------------------------------------------------------------------------------
csv_file_path = './ur5_dh.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Get DH Parameters (d, theta, r, alpha)
# dh_params[0:*] is parameter name of ['d', 'theta', 'r', 'alpha']
# dh_params[1:*] is world_joint paramerter
dh_params = df[['d', 'theta', 'r', 'alpha']].values
dh_params[:, 1] = np.radians(dh_params[:, 1])  # convert theta to radian
dh_params[:, 3] = np.radians(dh_params[:, 3])  # convert alpha to radian

# Get Joint names
joint_names = df['joint'].tolist()

# Config animation plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-1.0, 1.0])
ax.set_ylim([-1.0, 1.0])
ax.set_zlim([0, 1.0])
ax.view_init(elev=30, azim=-45)

fig2 = plt.figure(figsize=(10, 8))
ax2 = fig2.add_subplot(111)
ax2.set_xlabel('X')
ax2.set_ylabel('Z')
ax2.set_xlim([-1.0, 1.0])
ax2.set_ylim([-1.0, 1.0])

# coordinate system
def plot_coordinate_system(ax, T, scale=0.1):
    origin = T[:3, 3]
    x_axis = T[:3, 3] + scale * T[:3, 0]
    y_axis = T[:3, 3] + scale * T[:3, 1]
    z_axis = T[:3, 3] + scale * T[:3, 2]

    ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='red')
    ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='green')
    ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='blue')

# Update animation
def update(frame):
    ax.cla()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_zlim([0, 1.0])
    ax.view_init(elev=30, azim=-45)

    amplitude = np.radians(30)
    dh_params[2, 1] = amplitude * np.sin(frame)  # J1
    # dh_params[3, 1] = amplitude * np.sin(frame)  # J2
    dh_params[4, 1] = amplitude * np.sin(frame + np.pi/2)  # J3
    # dh_params[5, 1] = amplitude * np.sin(frame)  # J4
    # dh_params[6, 1] = amplitude * np.sin(frame)  # J5
    # dh_params[7, 1] = amplitude * np.sin(frame)  # J6

    T = np.eye(4)
    link_positions = []

    # Plotting the links and joints
    for i in range(len(dh_params)):
        d, theta, r, alpha = dh_params[i]

        # Transformation matrix
        Ti = np.array([
            [np.cos(theta), -np.sin(theta) * np.cos(alpha),  np.sin(theta) * np.sin(alpha), r * np.cos(theta)],
            [np.sin(theta),  np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), r * np.sin(theta)],
            [            0,                  np.sin(alpha),                  np.cos(alpha),                 d],
            [            0,                              0,                              0,                 1]
        ])

        T = np.dot(T, Ti)
        link_positions.append(T[:3, 3])

        # Plot the joint
        ax.scatter(T[0, 3], T[1, 3], T[2, 3], color='blue', s=100, label=joint_names[i])
        ax.text(T[0, 3], T[1, 3], T[2, 3], joint_names[i], color='black', fontsize=10, ha='center', va='top')

        # Plot the link
        if i > 0:
            ax.plot([link_positions[i-1][0], T[0, 3]],
                    [link_positions[i-1][1], T[1, 3]],
                    [link_positions[i-1][2], T[2, 3]],
                    color='blue')

        # plot system
        plot_coordinate_system(ax, T)

    # plot end effector
    ax.scatter(T[0, 3], T[1, 3], T[2, 3], color='red', s=100, label='End Effector')
    ax.legend()

    # plot position in X-Z plane
    ax2.plot([p[0] for p in link_positions], [p[2] for p in link_positions], color='gray', linestyle='--')
    ax2.set_title('End Effector Trajectory in X-Z Plane')

# Animation creation
frames = np.linspace(0, 2 * np.pi, 100)
ani = FuncAnimation(fig, update, frames=frames, interval=100)

# Displaying the figures
plt.tight_layout()
plt.show()