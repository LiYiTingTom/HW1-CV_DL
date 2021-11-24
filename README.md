# HW1 (Computer Vision and Deep Learning)

## Environment (use pipenv)
```bash
pipenv shell
pipenv install
```
## Question1-4
```bash
python3 main_window.py
```
## Question5
```bash
python3 q5_window.py
```
## Structure
### directories
1. data: 
    - Q1_Image
    - Q2_Image
    - Q3_Image
    - Q4_Image
### files
1. config.py: Define some Hyperparams for VGG16 model.
2. main_windows.py: Question1-4.
3. q5_window: Question5.
4. train.py: VGG16 training file.
5. utils.py: some utils for question5.

## Requirements
```Pipfile
[[source]]
url = "https://pypi.python.org/simple"
verify_ssl = true
name = "pypi"

[packages]
opencv-contrib-python = "*"
matplotlib = "*"
pyqt5 = "*"
opencv-python = "*"
numpy = "*"
scipy = "*"
tensorboardx = "*"
torchvision = "*"
torchsummary = "*"

[dev-packages]
ipykernel = "*"

[requires]
python_version = "3.8"
```
