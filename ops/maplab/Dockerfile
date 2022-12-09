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

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections
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

RUN sudo apt-get install -y python3-pip python3.7 python3-pip && \
  python3.7 -m pip install --upgrade pip && \
  python3.7 -m pip install --no-input opencv-python gnupg pycryptodomex numpy scipy scikit-learn && \
  python3.7 -m pip install --no-input open3d rospkg scikit-learn PyYaml && \
  python3.7 -m pip install --no-input rosbag roslz4 --extra-index-url https://rospypi.github.io/simple/

COPY map.sh /home/maplab_user/map.sh
COPY run_rovioli.sh /home/maplab_user/run_rovioli.sh
COPY maplab_console_script.yaml /home/maplab_user/maplab_console_script.yaml
COPY convert_to_autolabel.py /home/maplab_user/convert_to_autolabel.py
COPY create_rosbag.py /home/maplab_user/create_rosbag.py

ENTRYPOINT ["bash", "/home/maplab_user/map.sh"]

