FROM gcc:9.4

# https://github.com/Rikorose/gcc-cmake/blob/master/Dockerfile
ARG CMAKE_VERSION=3.23.1
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.sh \
      -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /usr/bin/cmake \
      && /tmp/cmake-install.sh --skip-license --prefix=/usr/bin/cmake \
      && rm /tmp/cmake-install.sh

ENV PATH="/usr/bin/cmake/bin:${PATH}"

RUN mkdir /dpf
WORKDIR /dpf
ADD data /dpf/data/
ADD include /dpf/include/
ADD src /dpf/src/
ADD test /dpf/test/
ADD CMakeLists.txt /dpf/

RUN mkdir build
WORKDIR /dpf/build 
RUN cmake ..
RUN cmake --build .
WORKDIR /dpf 