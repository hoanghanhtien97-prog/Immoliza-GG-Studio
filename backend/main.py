from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import joblib
import os

app = FastAPI(title="BelgiImmo Prediction API")

# --- CORS Middleware ---
# Allow all origins for development; restrict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Data Models ---

class PropertyData(BaseModel):
    zip_code: str
    province: str
    region: str
    type_of_property: str
    subtype_of_property: str
    type_of_sale: str
    number_of_rooms: int
    living_area: float
    fully_equipped_kitchen: str
    furnished: bool
    open_fire: bool
    terrace: bool
    terrace_area: Optional[float] = 0.0
    garden: bool
    garden_area: Optional[float] = 0.0
    surface_of_the_land: float
    number_of_facades: int
    swimming_pool: bool
    state_of_the_building: str

# --- Mappings ---

kitchen_scale = {
    'Not equipped': 0,
    'Partially equipped': 1,
    'Super equipped': 2,
    'Fully equipped': 3
}

building_scale = {
    'New': 4,
    'Under construction': 4,
    'Fully renovated': 3,
    'Excellent': 3,
    'Normal': 2,
    'To be renovated': 1,
    'To renovate': 1,
    'To restore': 1,
    'To demolish': 0
}

# --- Model Loading ---

# Placeholder for your trained model
# MODEL_PATH = "model.pkl"
# if os.path.exists(MODEL_PATH):
#     model = joblib.load(MODEL_PATH)
# else:
#     model = None

@app.get("/")
async def root():
    return {"message": "BelgiImmo Real Estate Prediction API is running"}

@app.get("/stats")
async def get_stats():
    """
    Get comprehensive market statistics for the dashboard.
    """
    # In a real application, these would be computed from the dataset
    return {
        "summary": {
            "avg_price": 345000,
            "transactions": 128000,
            "regions_tracked": 3
        },
        "top_expensive": {
            "total": [
                {"name": "Knokke-Heist", "price": "€845,000"},
                {"name": "Ixelles", "price": "€720,000"},
                {"name": "Uccle", "price": "€695,000"},
                {"name": "Sint-Martens-Latem", "price": "€680,000"},
                {"name": "Woluwe-Saint-Pierre", "price": "€650,000"},
            ],
            "per_m2": [
                {"name": "Ixelles", "price": "€4,850/m²"},
                {"name": "Saint-Gilles", "price": "€4,600/m²"},
                {"name": "Brussels City", "price": "€4,400/m²"},
                {"name": "Etterbeek", "price": "€4,350/m²"},
                {"name": "Knokke-Heist", "price": "€4,200/m²"},
            ]
        },
        "top_affordable": {
            "total": [
                {"name": "Hastière", "price": "€115,000"},
                {"name": "Colfontaine", "price": "€128,000"},
                {"name": "Quaregnon", "price": "€132,000"},
                {"name": "Viroinval", "price": "€135,000"},
                {"name": "Froidchapelle", "price": "€140,000"},
            ],
            "per_m2": [
                {"name": "Colfontaine", "price": "€1,050/m²"},
                {"name": "Quaregnon", "price": "€1,120/m²"},
                {"name": "Frameries", "price": "€1,180/m²"},
                {"name": "Dour", "price": "€1,220/m²"},
                {"name": "Charleroi", "price": "€1,250/m²"},
            ]
        },
        "regional_comparison": [
            {"name": "Brussels", "house": 485000, "apartment": 312000, "avgM2": 3200},
            {"name": "Flanders", "house": 340000, "apartment": 245000, "avgM2": 2600},
            {"name": "Wallonia", "house": 210000, "apartment": 185000, "avgM2": 1800},
        ],
        "scatter_data": [
            {"area": 80, "price": 250000, "region": "Flanders"},
            {"area": 120, "price": 380000, "region": "Flanders"},
            {"area": 150, "price": 450000, "region": "Flanders"},
            {"area": 200, "price": 580000, "region": "Flanders"},
            {"area": 60, "price": 280000, "region": "Brussels"},
            {"area": 90, "price": 420000, "region": "Brussels"},
            {"area": 130, "price": 550000, "region": "Brussels"},
            {"area": 180, "price": 720000, "region": "Brussels"},
            {"area": 100, "price": 180000, "region": "Wallonia"},
            {"area": 140, "price": 240000, "region": "Wallonia"},
            {"area": 190, "price": 310000, "region": "Wallonia"},
            {"area": 250, "price": 420000, "region": "Wallonia"},
        ],
        "market_distribution": {
            "House": [
                {"name": "Budget (<300k)", "value": 35, "color": "#E5E7EB"},
                {"name": "Mid-Range (300k-600k)", "value": 45, "color": "#9CA3AF"},
                {"name": "Premium (600k-1.2M)", "value": 15, "color": "#4B5563"},
                {"name": "Luxury (>1.2M)", "value": 5, "color": "#1F2937"},
            ],
            "Apartment": [
                {"name": "Budget (<300k)", "value": 55, "color": "#E5E7EB"},
                {"name": "Mid-Range (300k-600k)", "value": 35, "color": "#9CA3AF"},
                {"name": "Premium (600k-1.2M)", "value": 8, "color": "#4B5563"},
                {"name": "Luxury (>1.2M)", "value": 2, "color": "#1F2937"},
            ]
        },
        "currency": "EUR"
    }

@app.post("/predict")
async def predict(data: PropertyData):
    """
    Predict the price of a property based on its features.
    """
    try:
        # Convert Pydantic model to dictionary
        input_dict = data.dict()
        
        # Apply categorical mappings
        input_dict['fully_equipped_kitchen'] = kitchen_scale.get(data.fully_equipped_kitchen, 0)
        input_dict['state_of_the_building'] = building_scale.get(data.state_of_the_building, 2)
        
        # Convert to DataFrame (standard format for most ML models)
        df = pd.DataFrame([input_dict])
        
        # Perform prediction
        # if model:
        #     prediction = model.predict(df)[0]
        # else:
        #     # Mock prediction logic if model is not loaded
        #     prediction = 250000.0 + (data.living_area * 2500)
        
        # Mock result for demonstration
        prediction = 325000.0
        
        return {
            "status": "success",
            "prediction": round(float(prediction), 2),
            "currency": "EUR"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
