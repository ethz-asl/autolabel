#!/bin/bash

# Usage:
# ./setup-ubuntu.sh [arm64 | amd64]

set -e

apt-get update

packages=(\
    wget \
    dialog \
    debconf-utils \
    apt-utils \
    file \
    dpkg-dev \
    pkg-config \
    python3 \
    build-essential \
    git )

echo "apt-get install -y --no-install-recommends ${packages[@]}"
apt-get install -y --no-install-recommends ${packages[@]}

