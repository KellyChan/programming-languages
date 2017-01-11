wget https://aur.archlinux.org/packages/pc/pcl/pcl.tar.gz
tar -xzf pcl.tar.gz
cd pcl
makepkg
sudo pacman -U pcl*.xz
