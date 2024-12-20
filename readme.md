# Age Estimation API

This project provides an API for age estimation from images using a pre-trained deep learning model. The model predicts the age of a person in the input image.

## Features

- Accepts image input via a FastAPI endpoint.
- Utilizes a pre-trained `se_resnext50_32x4d` model for age estimation.
- Compatible with CPU and GPU environments.
- Includes a Dockerized setup for easy deployment.

---

## Installation

### Prerequisites

- Python 3.8 or later
- Docker and Docker Compose (optional for containerized deployment)

---

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone <repository_url>
   cd age-estimation-api
   ```

2. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download model weights:**
   Use the provided script to download the model weights.
   ```bash
   python download_weights.py
   ```
   Alternatively, use the bash script:
   ```bash
   ./download_weights.sh
   ```

4. **Run the API:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

---

### Dockerized Setup

1. **Build the Docker image:**
   ```bash
   docker build -t age-estimation-api .
   ```

2. **Run the Docker container:**
   ```bash
   docker run --gpus all -p 8000:8000 age-estimation-api
   ```

---

## Usage

### Endpoint

- **POST** `/predict/`
  - Accepts an image file (`multipart/form-data`).
  - Returns the predicted age of the person in the image.

#### Example cURL Command

```bash
curl -X POST http://localhost:8000/predict/ -F "file=@path_to_image.jpg"
```

#### Response

```json
{
  "age": 25
}
```

---

## Files

- `main.py`: FastAPI app for serving the API.
- `requirements.txt`: Python dependencies.
- `download_weights.py`: Python script to download pre-trained model weights.
- `download_weights.sh`: Bash script to download pre-trained model weights.
- `Dockerfile`: Docker configuration for the API.
- `pretrained.pth`: Pre-trained model weights (downloaded via script).

---

## Model

- **Architecture:** `se_resnext50_32x4d`
- **Source:** [Hugging Face Hub](https://huggingface.co/public-data/yu4u-age-estimation-pytorch/resolve/main/pretrained.pth)

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.

---

## Acknowledgments

- [Yu4u Age Estimation Repository](https://github.com/yu4u/age-estimation-pytorch)
- [Hugging Face Hub](https://huggingface.co)
