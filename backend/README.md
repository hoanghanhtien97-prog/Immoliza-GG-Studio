# BelgiImmo Prediction API

This is a FastAPI-based backend for the BelgiImmo real estate price prediction model.

## Features
- **Pydantic Data Validation**: Ensures all input data matches the required schema.
- **Categorical Mapping**: Automatically maps text-based features (like kitchen equipment and building state) to numerical scales.
- **CORS Support**: Pre-configured to allow requests from the React frontend.
- **Dockerized**: Includes a `Dockerfile` for easy deployment.

## Getting Started

### Prerequisites
- Python 3.11+
- pip

### Installation
1. Navigate to the backend directory:
   ```bash
   cd backend
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the API
Start the server using uvicorn:
```bash
uvicorn main:app --reload
```
The API will be available at `http://localhost:8000`.

### API Documentation
FastAPI automatically generates interactive documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## Deployment

### Using Docker
1. Build the image:
   ```bash
   docker build -t belgiimmo-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8000:8000 belgiimmo-api
   ```

## Model Integration
To use your actual trained model:
1. Place your model file (e.g., `model.pkl`) in the `/backend` directory.
2. Uncomment the model loading logic in `main.py`.
3. Update the `predict` function to use `model.predict(df)`.
