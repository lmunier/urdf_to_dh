# URDF to DH Parameterization

v0.0.2

Sometimes people want DH.

## Documentation

Check out the [documentation](https://mcevoyandy.github.io/urdf_to_dh/index.html) for more details on how the conversion is done and how to use this package.

## Running the node

```sh: terminal
ros2 run urdf_to_dh generate_dh --ros-args -p urdf_file:="<path_to_my_urdf>"
# ex.)
ros2 run urdf_to_dh generate_dh --ros-args -p urdf_file:=/home/ubuntu/dev_ws/src/urdf_to_dh/urdf/ur5.urdf
```

## DH parameters confirm
To check the robot link, do the following:

```sh: terminal
python3 view_trajectory_dh.py
```

To check the trajectory in 3D animation, do the following:

```sh: terminal
python3 view_trajectory3d_dh.py
```

![urdf_to_dh_sample](./gif/urdf_to_dh_sample.gif)
