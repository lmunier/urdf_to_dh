# Handle ros distro
ARG ROS_DISTRO=humble
ARG ROS_IMAGE=${ROS_DISTRO}-desktop

FROM osrf/ros:${ROS_IMAGE} AS ros-ws

ARG ROS_DISTRO
ARG USER=root
ARG HOME=/home/${USER}

ENV DEBIAN_FRONTEND noninteractive
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES video,compute,utility
ENV LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

# Tell docker we want to use bash instead of sh in general
SHELL ["/bin/bash", "-c"]

# create and configure a new user
RUN apt update --fix-missing && apt upgrade -y && apt clean
RUN useradd -m ${USER} && echo "${USER}:${USER}passwd" | chpasswd && adduser ${USER} sudo
RUN usermod -a -G dialout ${USER}

WORKDIR ${HOME}
RUN chown -R ${USER}:${USER} ${HOME}

# Install ubuntu libraries
RUN apt update --fix-missing && apt upgrade -y && apt clean
RUN apt install -y \
    git \
    python3 \
    python3-pip

# Install ROS packages dependencies
RUN apt update --fix-missing && apt upgrade -y && apt clean
RUN apt install -y \
    ros-${ROS_DISTRO}-joint-state-publisher-gui \
    ros-${ROS_DISTRO}-xacro

# Install urdf_to_dh parameters dependencies
RUN pip3 install --upgrade pip
RUN pip3 install --upgrade \
    pandas \
    anytree \
    tabulate \
    scipy

RUN echo "source /opt/ros/${ROS_DISTRO}/setup.bash" >> ~/.bashrc

WORKDIR ${HOME}/ros_ws
CMD [ "bash" ]