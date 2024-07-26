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
from math import pow

import urdf_to_dh.kinematics_helpers as kh
import urdf_to_dh.urdf_helpers as uh


PRECISION = 8  # Number of decimal places to round to or compare to be close enough
EPSILON = pow(10, -PRECISION)  # Tolerance for floating point comparisons


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
        self.root_link = None
        self.reference_axis = None

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
            if n.is_root:
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

        # Define axis for fixed joints
        for n in LevelOrderIter(self.root_link):
            if n.type == 'joint':
                joint = self.urdf_joints[n.id]

                if joint['type'] == 'fixed' and joint['axis'] is None:
                    joint['axis'] = uh.get_axis(n, self.urdf_joints)

                if self.reference_axis is None:
                    self.reference_axis = uh.get_reference_axis(
                        joint
                    )
                    print(f"Reference Axis: {self.reference_axis}\n")

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

                joint_axis, parent_joint_axis = self.get_joint_and_parent_axis(
                    n
                )

                tf = self.urdf_links[n.id]['rel_tf']
                joint_axis_in_parent = kh.normalize(tf[0:3, 0:3] @ joint_axis)

                common_normal = self.get_common_normal(
                    parent_joint_axis, joint_axis_in_parent, tf,
                )

                # DH parameters
                theta_val, d_val, a_val, alpha_val = 0.0, 0.0, 1.0, 1.0
                # a_val is take positive to determine its sign if theta = PI is to be changed as theta = 0

                theta_val = np.arccos(
                    np.dot(self.reference_axis, common_normal)
                )

                theta_val, a_val, alpha_val, common_normal = self.adjust_dh_sign(
                    theta_val, a_val, alpha_val, common_normal
                )
                self.reference_axis = kh.inv_tf(tf)[0:3, 0:3] @ common_normal

                alpha_val *= np.arccos(
                    np.dot(parent_joint_axis, joint_axis_in_parent)
                )
                alpha_val = 0 if np.abs(alpha_val) < EPSILON else alpha_val

                # 'd'
                d_val = np.dot(tf[0:3, 3], parent_joint_axis)

                # 'a' or 'r'
                if np.abs(alpha_val) < EPSILON or np.abs(alpha_val) - np.pi < EPSILON:
                    a_val *= np.sqrt(np.linalg.norm(tf[0:3, 3])**2 - d_val**2)
                else:
                    a_val *= np.dot(tf[0:3, 3], common_normal)

                # Round and adjust values
                alpha_val, d_val = self.adjust_dh_sign(alpha_val, d_val)
                d_val = 0 if abs(d_val) < EPSILON else round(d_val, PRECISION)
                a_val = 0 if abs(a_val) < EPSILON else round(a_val, PRECISION)

                self.urdf_links[n.id]['dh_found'] = True
                self.urdf_joints[n.id] = np.array(
                    [d_val, theta_val, a_val, alpha_val]
                )

    def get_joint_and_parent_axis(self, node: AnyNode) -> tuple:
        """ Get the joint and parent axis.

        Args:
            node: The joint node to get the joint and parent axis from.

        Returns:
            joint_axis: The axis of the joint.
            parent_joint_axis: The axis of the parent joint.
        """
        parent_joint_axis = None
        joint_axis = kh.normalize(
            self.urdf_joints[node.parent.id]['axis']
        )

        try:
            if self.urdf_joints[node.parent.parent.parent.id]['type'] in ["fixed", None]:
                parent_joint_axis = joint_axis
            else:
                parent_joint_axis = kh.normalize(
                    self.urdf_joints[node.parent.parent.parent.id]['axis']
                )
        except:
            parent_joint_axis = joint_axis

        return joint_axis, parent_joint_axis

    def get_common_normal(
        self, parent_joint_axis: np.ndarray, joint_axis_in_parent: np.ndarray,
        tf: np.ndarray
    ) -> np.ndarray:
        """ Get the common normal.

        Args:
            parent_joint_axis: The axis of the parent joint.
            joint_axis_in_parent: The axis of the joint in the parent joint.
            tf: The transformation matrix.

        Returns:
            common_normal: The common normal between the joint and the parent joint.
        """
        common_normal = kh.normalize(
            np.cross(parent_joint_axis, joint_axis_in_parent)
        )

        if np.linalg.norm(common_normal) < EPSILON:
            # Coincident revolute axis
            if np.linalg.norm(np.cross(parent_joint_axis, tf[0:3, 3])) < EPSILON:
                common_normal = kh.inv_tf(
                    tf
                )[0:3, 0:3] @ kh.normalize(self.reference_axis)

            # Parallel revolute axis
            else:
                common_normal = kh.normalize(np.cross(
                    np.cross(parent_joint_axis, tf[0:3, 3]),
                    parent_joint_axis
                ))

        return common_normal

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
                    ] + self.urdf_joints[urdf_node.id].tolist()
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

    def adjust_dh_sign(self, angle: float, *args: list) -> tuple:
        """ Adjust the sign of the value based on the angle. """
        adapted_values = (angle, ) + args

        if np.cos(angle) < 0 and np.abs(np.cos(angle)) > EPSILON:
            angle -= np.pi
            adapted_values = (angle,)

            for v in args:
                if type(v) in [int, float]:
                    v = -v if abs(v) > EPSILON else 0
                elif type(v) in [np.int32, np.float32, np.int64, np.float64]:
                    v = -v if np.abs(v) > EPSILON else 0
                elif type(v) is np.ndarray:
                    v = -v if np.linalg.norm(v) > EPSILON else 0
                else:
                    print(f'ERROR: Type {type(v)} not managed.')

                adapted_values += (v,)

        return adapted_values


def main():
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
