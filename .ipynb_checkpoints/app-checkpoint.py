import gradio as gr
import pickle
import numpy as np

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Optional mappings (if model expects encoded values)
make_mapping = {
    "alfa-romero": 0, "audi": 1, "bmw": 2, "chevrolet": 3, "dodge": 4,
    "honda": 5, "isuzu": 6, "jaguar": 7, "mazda": 8, "mercedes-benz": 9,
    "mitsubishi": 10, "nissan": 11, "peugot": 12, "plymouth": 13,
    "porsche": 14, "renault": 15, "saab": 16, "subaru": 17, "toyota": 18,
    "volkswagen": 19, "volvo": 20
}

fuel_mapping = {"gas": 0, "diesel": 1}
body_mapping = {"convertible": 0, "hatchback": 1, "sedan": 2, "wagon": 3, "hardtop": 4}
drive_mapping = {"fwd": 0, "rwd": 1, "4wd": 2}
engine_loc_mapping = {"front": 0, "rear": 1}
engine_type_mapping = {"dohc": 0, "ohcv": 1, "ohc": 2, "rotor": 3, "l": 4}

# Prediction function
def predict_price(symboling, normalized_losses, make, fuel_type, body_style,
                  drive_wheels, engine_location, width, height, engine_type,
                  engine_size, horsepower, city_mpg, highway_mpg):
    try:
        # Handle missing or unknown values
        normalized_losses = float(normalized_losses) if normalized_losses != "?" else 100.0

        # Encode categorical features
        make_code = make_mapping.get(make, -1)
        fuel_code = fuel_mapping.get(fuel_type, -1)
        body_code = body_mapping.get(body_style, -1)
        drive_code = drive_mapping.get(drive_wheels, -1)
        engine_loc_code = engine_loc_mapping.get(engine_location, -1)
        engine_type_code = engine_type_mapping.get(engine_type, -1)

        # Combine features
        features = np.array([
            symboling, normalized_losses, make_code, fuel_code, body_code,
            drive_code, engine_loc_code, width, height, engine_type_code,
            engine_size, horsepower, city_mpg, highway_mpg
        ]).reshape(1, -1)

        prediction = model.predict(features)
        return f"Estimated Price: ₹{round(prediction[0], 2):,}"
    except Exception as e:
        return f"Error: {str(e)}"

# Gradio UI
demo = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Symboling"),
        gr.Textbox(label="Normalized Losses (use '?' if unknown)"),
        gr.Dropdown(label="Make", choices=list(make_mapping.keys()), value="alfa-romero"),
        gr.Dropdown(label="Fuel Type", choices=list(fuel_mapping.keys()), value="gas"),
        gr.Dropdown(label="Body Style", choices=list(body_mapping.keys()), value="convertible"),
        gr.Dropdown(label="Drive Wheels", choices=list(drive_mapping.keys()), value="rwd"),
        gr.Dropdown(label="Engine Location", choices=list(engine_loc_mapping.keys()), value="front"),
        gr.Number(label="Width"),
        gr.Number(label="Height"),
        gr.Dropdown(label="Engine Type", choices=list(engine_type_mapping.keys()), value="dohc"),
        gr.Number(label="Engine Size"),
        gr.Number(label="Horsepower"),
        gr.Number(label="City MPG"),
        gr.Number(label="Highway MPG")
    ],
    outputs="text",
    title="🚗 Full Vehicle Price Predictor",
    description="Enter complete vehicle specs to estimate price using ML model."
)

demo.launch()