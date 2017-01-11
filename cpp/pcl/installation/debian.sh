sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key 19274DEF
sudo echo "deb http://ppa.launchpad.net/v-launchpad-jochen-sprickerhof-de/pcl/ubuntu maverick main" >> /etc/apt/sources.list
sudo apt-get update
sudo apt-get install libpcl-all
