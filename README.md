# CS4243 Mini Project: CAPTCHA Recognition

## Overview
This repository contains the implementation for the CS4243 Mini Project on CAPTCHA recognition using deep learning techniques. The objective is to develop a model capable of recognizing text-based CAPTCHAs with high accuracy.

## Project Structure

## Setup Instructions
### Prerequisites
- Python 3.8+
- Virtual environment (optional but recommended)
- GPU (recommended for faster training)

### Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/cs4243-captcha.git
   cd cs4243-captcha
   ```
2. (Optional) Create a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

## Usage
### Preprocessing
```sh
python src/preprocessing.py
```

### Training the Model
```sh
python src/train.py --epochs 50 --batch_size 32
```

### Evaluating the Model
```sh
python src/evaluate.py --model_path models/best_model.pth
```

## Contributing
Feel free to fork this repository and submit pull requests with improvements!

## License
This project is for educational purposes only.

## Contact
For any issues, please open an issue in this repository.

