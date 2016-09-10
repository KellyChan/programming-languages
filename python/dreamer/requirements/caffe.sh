# Ubuntu 14.04
#--------------------------------------------------------------------------#
##### deepdream

##### python
sudo apt-get install python-dev

##### Pillow
sudo apt-get install libtiff5-dev libjpeg8-dev zlib1g-dev \
libfreetype6-dev liblcms2-dev libwebp-dev tcl8.6-dev tk8.6-dev python-tk

##### Scipy
sudo apt-get install libblas-dev liblapack-dev libatlas-base-dev gfortran

#--------------------------------------------------------------------------#
##### Caffe
# reference: https://gist.github.com/titipata/f0ef48ad2f0ebc07bcb9
# check version
lspci | grep -i nvidia
uname -m && cat /etc/*release
gcc --version

# CUDA
cd /tmp
wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1404/x86_64/cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1404_6.5-14_amd64.deb
sudo apt-get update
sudo apt-get install cuda
sudo reboot

# BLAS
sudo apt-get install libopenblas-dev

# OpenCV
sudo apt-get install git unzip
cd /tmp
wget https://raw.githubusercontent.com/jayrambhia/Install-OpenCV/master/Ubuntu/2.4/opencv2_4_9.sh
sudo chmod +x opencv2_4_9.sh
./opencv2_4_9.sh

# Anaconda
cd /tmp
wget http://09c8d0b2229f813c1b93-c95ac804525aac4b6dba79b00b39d1d3.r79.cf1.rackcdn.com/Anaconda-2.1.0-Linux-x86_64.sh
sudo bash Anaconda-2.1.0-Linux-x86.sh
# location: /usr/local/anaconda

# Boost
sudo apt-get install libboost-all-dev 

# Caffe Dependencies
# reference: http://caffe.berkeleyvision.org/install_apt.html
sudo apt-get install libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev libboost-all-dev libhdf5-serial-dev
sudo apt-get install libgflags-dev libgoogle-glog-dev liblmdb-dev protobuf-compiler

# protobuf
sudo apt-get install python-pip
sudo pip install protobuf

# Caffe
cd /usr/local/
sudo git clone https://github.com/BVLC/caffe
cd caffe/
sudo cp Makefile.config.example Makefile.config
# modify Makefile.config if needed
###############################################
# CPU-only switch (uncomment to build without GPU support).
CPU_ONLY := 1
###############################################
sudo make all
sudo make test
# sudo vim ~/.bashrc            # update paths
# source ~/.bashrc
make runtest

# fixed Anaconda issue
sudo rm /usr/local/anaconda/lib/libm.*

sudo make pycaffe
sudo make distribute
# make dir for custom python modules, install caffe
sudo mkdir ~/pycaffe
mv distribute/python/caffe ~/pycaffe
###############################################################
# set PYTHONPATH (this should go in your .bashrc or whatever
export PYTHONPATH=${HOME}/pycaffe:$PYTHONPATH
###############################################################
source ~/.bashrc

sudo vim ~/.bashrc
####################################################################
# CUDA
export PATH=/usr/local/cuda-7.5/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:$LD_LIBRARY_PATH
export PATH
# Anaconda
export PATH=/usr/local/anaconda/bin:$PATH
# Caffe Root
export CAFFE_ROOT=/usr/local/caffe
####################################################################
