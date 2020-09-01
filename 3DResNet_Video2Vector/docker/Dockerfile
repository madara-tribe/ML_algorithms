# cuda on ubuntu 16.04
FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Yuya Yoshikawa <yoshikawa@stair.center>

# ffmpeg
RUN \
    apt-get update -y && \
    apt-get install software-properties-common -y && \
    apt-get update -y && \
    apt-get install -y ffmpeg libav-tools x264 x265

# python 2.7 and opencv 3.
RUN \ 
    apt-get update && apt-get upgrade -y && \
    apt-get install build-essential cmake git pkg-config -y && \
    apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev -y && \
    apt-get install libgtk2.0-dev -y && \
    apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev -y && \
    apt-get install libatlas-base-dev gfortran -y && \    
    apt-get install --assume-yes build-essential cmake git && \
    apt-get install --assume-yes build-essential pkg-config unzip ffmpeg qtbase5-dev python-dev && \
    apt-get install --assume-yes libopencv-dev libgtk-3-dev libdc1394-22 libdc1394-22-dev libjpeg-dev libpng12-dev libtiff5-dev libjasper-dev && \
    apt-get install --assume-yes libavcodec-dev libavformat-dev libswscale-dev libxine2-dev libgstreamer0.10-dev libgstreamer-plugins-base0.10-dev && \
    apt-get install --assume-yes libv4l-dev libtbb-dev libfaac-dev libmp3lame-dev libopencore-amrnb-dev libopencore-amrwb-dev libtheora-dev && \
    apt-get install --assume-yes libvorbis-dev libxvidcore-dev v4l-utils && \
    apt-get install libhdf5-dev -y && \
    apt-get install python-pip -y && \
    pip install --upgrade pip && \
    pip install virtualenv virtualenvwrapper && \
    rm -rf ~/.cache/pip && \
    apt-get install python2.7-dev -y && \
    pip install ipython numpy scipy matplotlib scikit-learn && \
    apt-get autoclean && apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN \
    cd ~ && \
    git clone https://github.com/Itseez/opencv.git && \
    cd opencv && \
    pwd && \
    git checkout 3.2.0 && \
    cd ~ && \
    git clone https://github.com/Itseez/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout 3.2.0 && \
    cd /root/opencv && \
    mkdir build && \
    cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D INSTALL_C_EXAMPLES=OFF \
        -D INSTALL_PYTHON_EXAMPLES=ON \
	-D WITH_FFMPEG=ON \
	-D WITH_CUDA=OFF \
	-D WITH_TBB=ON \
	-D WITH_V4L=ON \
	-D WITH_QT=ON \
	-D WITH_OPENGL=ON \
        -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_contrib/modules \
        -D BUILD_EXAMPLES=ON .. && \
    cd ~/opencv/build && \
    make -j $(nproc) && \
    make install && \
    ldconfig && \
    cp ~/opencv/build/lib/cv2.so /usr/local/lib/python2.7/dist-packages/

# chainer 1.24
RUN pip install cupy==2.0.0a1 chainer==1.24

# ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
