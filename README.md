# speckchip_record
## This program allows speck chip to read frames from event camera and saves a dataset. 

### Steps of using this program:

0. Make a file in ```/etc/udev/rules.d``` named ```60-synsense.rules``` which contains the following:
```
SUBSYSTEM=="usb", ATTR{idVendor}=="04b4", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="152a", MODE="0666"
SUBSYSTEM=="usb", ATTR{idVendor}=="337d", MODE="0666"
```
Followed by executing
```
# udevadm control --reload-rules
# udevadm trigger
```
As introduced by [Samna ducomentation](https://synsense-sys-int.gitlab.io/samna/install.html).

1. Use Anaconda to create a virtual environment 'SpeckChip0' with ```conda env create -f environment.yml```; Then ```conda activate SpeckChip0```.
   
2. ```git clone https://github.com/niart/speckchip_record.git``` and ```cd speckchip_record```;
 
3. ```python main.py``` to play the camera, or ```python record.py``` to record a dataset.
