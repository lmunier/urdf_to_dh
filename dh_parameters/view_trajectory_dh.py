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
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------------
# Read urdf_to_dh parameters from csv file.
# --------------------------------------------------------------------------------
csv_file_path = './ur5_dh.csv'

# Read CSV file
df = pd.read_csv(csv_file_path)

# Get DH Parameters (d, theta, r, alpha)
dh_params = df[['d', 'theta', 'r', 'alpha']].values
dh_params[:, 1] = np.radians(dh_params[:, 1])  # convert theta to radian
dh_params[:, 3] = np.radians(dh_params[:, 3])  # convert alpha to radian

# Get Joint names
joint_names = df['joint'].tolist()

# Create figure and 3D axis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-0.5, 0.5])
ax.set_ylim([-0.5, 0.5])
ax.set_zlim([0, 1.0])
ax.view_init(elev=30, azim=-45)

# Transformation matrices
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

# Plot end-effector
ax.scatter(T[0, 3], T[1, 3], T[2, 3], color='red', s=100, label='End Effector')
ax.legend()

plt.show()
