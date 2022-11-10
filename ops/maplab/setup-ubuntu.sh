#!/bin/bash

# Usage:
# ./setup-ubuntu.sh [arm64 | amd64]

set -e

apt-get update

apt-get install wget -y

apt-get update

packages=(\
	file \
	dpkg-dev \
	pkg-config \
	python3 \
	build-essential \
	git )

if [ "$arch" = "amd64" ]; then
    packages+=(libopencv-dev:$arch)
fi

echo "apt-get install -y --no-install-recommends ${packages[@]}"
apt-get install -y --no-install-recommends ${packages[@]}

