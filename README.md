# Four Tank System LSTM Predictor

This Streamlit application predicts liquid heights in a four-tank system based on pump speeds using an LSTM neural network.

## Features

- Interactive pump speed controls (v1 and v2)
- Set initial tank heights
- Real-time prediction visualization
- Multiple visualization options (combined plot, individual tanks)
- Data table with download functionality
- No LSTM parameters exposed to users

## Setup Instructions

### 1. Train and Save the Model

First, ensure you have your `four_tank_data.csv` file in the same directory, then run:

```bash
python train_and_save_model.py
```

This will:
- Train the LSTM model on your data
- Save the model as `lstm_four_tank_model.h5`
- Save the scalers and parameters as `model_artifacts.pkl`

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Locally

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Deployment Options

### Option 1: Streamlit Community Cloud (Recommended)

1. Push your code to GitHub with these files:
   - `app.py`
   - `requirements.txt`
   - `lstm_four_tank_model.h5`
   - `model_artifacts.pkl`

2. Go to [share.streamlit.io](https://share.streamlit.io)

3. Sign in with GitHub

4. Click "New app"

5. Select your repository, branch, and `app.py`

6. Click "Deploy"

**Note**: Make sure the model files (`lstm_four_tank_model.h5` and `model_artifacts.pkl`) are in your GitHub repository.

### Option 2: Heroku

1. Create a `Procfile`:
```
web: sh setup.sh && streamlit run app.py
```

2. Create a `setup.sh`:
```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: AWS/Azure/GCP

Use Docker with the following Dockerfile:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py"]
```

Build and deploy:
```bash
docker build -t tank-predictor .
docker run -p 8501:8501 tank-predictor
```

## File Structure

```
.
├── app.py                          # Streamlit application
├── train_and_save_model.py         # Model training script
├── requirements.txt                # Python dependencies
├── lstm_four_tank_model.h5         # Trained model (generated)
├── model_artifacts.pkl             # Scalers and parameters (generated)
├── four_tank_data.csv             # Training data (your file)
└── README.md                       # This file
```

## Usage

1. **Set Pump Speeds**: Use the sliders in the sidebar to set v1 and v2
2. **Set Initial Heights**: Input the starting heights for all four tanks
3. **Choose Prediction Steps**: Select how many time steps to predict
4. **Click Predict**: Click the "Predict Heights" button to generate predictions
5. **View Results**: Explore the predictions in three different tabs:
   - Time Series Plot: All tanks on one graph
   - Individual Tanks: Separate plots for each tank
   - Data Table: Numerical results with download option

## Model Details

- **Architecture**: 2-layer LSTM with dropout
- **Input Features**: Pump speeds (v1, v2) and previous tank heights (h1-h4)
- **Output**: Predicted heights for all four tanks
- **Sequence Length**: 20 time steps
- **Training**: Early stopping based on validation loss

## Notes

- The model uses MinMax scaling for inputs and outputs
- Predictions are made iteratively, feeding predictions back as inputs
- All LSTM internal parameters are hidden from the user interface
- The app focuses on providing an intuitive control interface

## Troubleshooting

**Model file not found**: Make sure you run `train_and_save_model.py` first to generate the model files.

**Memory errors**: Large prediction steps (>200) may cause memory issues. Try reducing the prediction steps.

**Deployment size**: Model files can be large. For GitHub, ensure files are under 100MB or use Git LFS.

## License

MIT License - Feel free to modify and use for your projects.
