#!/bin/bash

#!/bin/bash

# Step 1: Update package lists
apt update -y

# Step 2: Upgrade installed packages
apt upgrade -y
apt install python3-pip unzip

# Step 3: Clone the specified repository and checkout the branch
git clone --branch gated-yolo-merge https://github.com/hectorandac/Gated-YOLOv6.git

cd Gated-YOLOv6

# Step 4: Install required Python packages
pip3 install cython
pip3 install numpy==1.24.3
pip3 install tensorboard
pip3 install matplotlib
pip3 install psutil
pip3 install torchvision
pip3 install pycocotools==2.0.6
pip3 install -r requirements.txt

# Step 5: Load the VOC dataset
cd data/scripts
./get_mnist.sh
./get_voc.sh

echo "Setup completed successfully."