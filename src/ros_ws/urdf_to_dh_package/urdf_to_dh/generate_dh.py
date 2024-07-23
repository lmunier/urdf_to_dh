# Copyright 2024 Louis Munier and Durgesh Salunkhe.
# Copyright 2024 Takumi Asada.
# Copyright 2020 Andy McEvoy.
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

import rclpy
import rclpy.node
import numpy as np
import os
import pandas as pd
import pprint
import xml.etree.ElementTree as ET

from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from anytree import AnyNode, LevelOrderIter, RenderTree
from scipy.spatial.transform import Rotation as R
from math import atan2, sqrt

import urdf_to_dh.kinematics_helpers as kh
import urdf_to_dh.geometry_helpers as gh
import urdf_to_dh.urdf_helpers as uh
import urdf_to_dh.marker_helpers as mh
from urdf_to_dh.math_basics import *


EPSILON = 1e-6  # Tolerance for floating point comparisons


class GenerateDhParams(rclpy.node.Node):

    def __init__(self):
        super().__init__('generate_dh_param_node')
        default_urdf = os.path.join(
            os.getcwd(), 'urdf_to_dh_package/urdf/random.urdf'
        )

        self.get_logger().info('Initializing...')
        self.declare_parameter(
            'urdf_file',
            default_urdf,
            ParameterDescriptor(type=ParameterType.PARAMETER_STRING)
        )

        self.urdf_joints = {}
        self.urdf_links = {}
        self.urdf_tree_nodes = []
        self.dh_params = {}
        self.root_link = None

        self.urdf_file = self.get_parameter(
            'urdf_file'
        ).get_parameter_value().string_value

        self.get_logger().info('URDF file = %s' % self.urdf_file)

    def parse_urdf(self):
        """ Parse the URDF file and create a tree structure. """
        # Get the root of the URDF and extract all of the joints
        urdf_root = uh.get_urdf_root(self.urdf_file)

        # Parse all links first and add to tree
        for child in urdf_root:
            if child.tag == 'link':
                self.urdf_links[child.get('name')] = {
                    'rel_tf': np.eye(4),
                    'dh_found': False
                }

                link_node = AnyNode(
                    id=child.get('name'),
                    parent=None,
                    children=None,
                    type='link'
                )
                self.urdf_tree_nodes.append(link_node)

        # Parse all joints and add to tree
        for child in urdf_root:
            if child.tag == 'joint':
                joint_name, joint_data = uh.process_joint(child)
                self.urdf_joints[joint_name] = joint_data

                joint_node = AnyNode(
                    id=joint_name,
                    parent=None,
                    children=None,
                    type='joint'
                )

                # Find parent and child link
                for n in self.urdf_tree_nodes:
                    if n.id == joint_data['parent']:
                        joint_node.parent = n
                    if n.id == joint_data['child']:
                        n.parent = joint_node

                self.urdf_tree_nodes.append(joint_node)

        # Find root link
        num_nodes_no_parent = 0
        for n in self.urdf_tree_nodes:
            if n.parent == None:
                num_nodes_no_parent += 1
                self.root_link = n

        if num_nodes_no_parent == 1:
            # Root link DH will be identity, set dh_found = True
            self.urdf_links[self.root_link.id]['dh_found'] = True

            print(f"\nURDF Tree:")
            for pre, _, n in RenderTree(self.root_link):
                print(f"{pre}{n.id}")

            print(f"\nJoint Info:")
            pprint.pprint(self.urdf_joints)
            print(f"\n")
        else:
            print("Error: Should only be one root link")

    def calculate_tfs(self):
        """ Calculate the transformation matrices for each link in both the local and the world frame. """
        print("Calculate world tfs:")

        for n in LevelOrderIter(self.root_link):
            if n.type == 'link' and n.parent is not None:
                print(f"- Get tf from {n.parent.parent.id} to {n.id}")

                xyz = self.urdf_joints[n.parent.id]['xyz']
                rpy = self.urdf_joints[n.parent.id]['rpy']

                # Compute relative tf
                tf = np.eye(4)
                tf[0:3, 0:3] = kh.get_extrinsic_rotation(rpy)
                tf[0:3, 3] = xyz
                self.urdf_links[n.id]['rel_tf'] = tf

    def compute_dh_params(self):
        for n in LevelOrderIter(self.root_link):
            if n.type == 'link' and n.parent is not None:
                # Condition to not compute DH params that are already known
                if self.urdf_links[n.id]['dh_found']:
                    continue

                joint_axis = kh.normalize(
                    self.urdf_joints[n.parent.id]['axis']
                )

                try:
                    if self.urdf_joints[n.parent.parent.parent.id]['type'] in ["fixed", None]:
                        parent_joint_axis = joint_axis
                    else:
                        parent_joint_axis = kh.normalize(
                            self.urdf_joints[n.parent.parent.parent.id]['axis']
                        )
                except:
                    parent_joint_axis = joint_axis

                # Get common normal
                tf = self.urdf_links[n.id]['rel_tf']
                joint_axis_in_parent = kh.normalize(tf[0:3, 0:3] @ joint_axis)
                common_normal = kh.normalize(
                    np.cross(joint_axis_in_parent, parent_joint_axis)
                )

                # DH parameters
                alpha_val = np.arccos(
                    np.dot(parent_joint_axis, joint_axis_in_parent)
                )

                d_val = np.dot(tf[0:3, 3], parent_joint_axis)

                a_val = 0
                if abs(alpha_val) < EPSILON or abs(alpha_val) - np.pi < EPSILON:
                    # TODO(lmunier) solve sign
                    a_val = np.sqrt(np.linalg.norm(tf[0:3, 3])**2 - d_val**2)
                else:
                    a_val = np.dot(tf[0:3, 3], common_normal)

                self.urdf_links[n.id]['dh_found'] = True
                self.dh_params[n.id] = [d_val, 0, a_val, alpha_val]

    def display_dh_params(self):
        """ Calculate the DH parameters for each joint. """
        robot_dh_params = []

        for urdf_node in LevelOrderIter(self.root_link):
            if urdf_node.type == 'link' and urdf_node.parent is not None:
                robot_dh_params.append(
                    [
                        urdf_node.parent.id,
                        urdf_node.parent.parent.id,
                        urdf_node.id
                    ] + self.dh_params[urdf_node.id]
                )

        pd_frame = pd.DataFrame(
            robot_dh_params,
            columns=[
                'joint', 'parent', 'child', 'd', 'theta', 'r', 'alpha'
            ]
        )
        pd_frame['theta'] = np.degrees(pd_frame['theta'])
        pd_frame['alpha'] = np.degrees(pd_frame['alpha'])

        base_filename = os.path.splitext(os.path.basename(self.urdf_file))[0]
        save_dir = os.path.join(
            os.getcwd(), 'urdf_to_dh_package/dh_parameters'
        )
        csv_file_path = os.path.join(save_dir, f"{base_filename}_dh.csv")
        markdown_file_path = os.path.join(save_dir, f"{base_filename}_dh.md")

        # Save CSV file.
        print("\nDH Parameters: (csv)")
        pd_frame.to_csv(csv_file_path, index=False)
        print(pd_frame.to_csv())

        # Save Markdown file.
        print("\nDH Parameters: (markdown)")
        with open(markdown_file_path, 'w') as file:
            file.write(pd_frame.to_markdown(index=False))

        print(pd_frame.to_markdown())


def main():
    print('Starting GenerateDhParams Node...')
    rclpy.init()
    node = GenerateDhParams()
    node.parse_urdf()
    node.calculate_tfs()
    node.compute_dh_params()
    node.display_dh_params()

    try:
        rclpy.spinOnce(node)
    except:
        pass

    rclpy.shutdown()


if __name__ == '__main__':
    main()
