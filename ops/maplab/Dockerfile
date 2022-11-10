FROM ros:melodic-ros-base-bionic

# Tell Ubuntu non-interactive install
ARG DEBIAN_FRONTEND=noninteractive

RUN echo 'root:docker' | chpasswd

RUN adduser --disabled-password --gecos '' maplab_user && chown maplab_user /home/maplab_user
RUN adduser maplab_user sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility,graphics,video,display

WORKDIR /root

ADD setup-ubuntu.sh /root

RUN ./setup-ubuntu.sh ${ARCH}

RUN apt-get install -y ca-certificates
RUN update-ca-certificates

ENV DISPLAY :1

USER maplab_user

ADD install.sh /home/maplab_user
RUN mkdir -p /home/maplab_user/ws/src

WORKDIR /home/maplab_user/ws

COPY maplab /home/maplab_user/ws/src/maplab

RUN sudo chown -R maplab_user /home/maplab_user/ws/src && /home/maplab_user/install.sh

COPY map.sh /home/maplab_user/map.sh
COPY maplab_console_script.yaml /home/maplab_user/maplab_console_script.yaml
RUN sudo apt-get install -y python3.7 python3-pip python3.7-dev python3-yaml && \
  python3.7 -m pip install cython numpy open3d rospkg gnupg pycryptodomex scikit-learn
COPY convert_to_autolabel.py /home/maplab_user/convert_to_autolabel.py

ENTRYPOINT ["bash", "/home/maplab_user/map.sh"]



