# 🔥 Wildfire Severity Prediction App

A stunning, AI-powered web application for predicting wildfire severity using machine learning.

## 🌟 Features

- **Beautiful Modern UI** with gradient backgrounds and glassmorphism effects
- **Multi-Page Navigation** (Home, Prediction, Analytics, About)
- **Real-Time Predictions** using trained ML models
- **Interactive Visualizations** with Plotly charts
- **Severity Classifications** (Minor, Moderate, Severe)
- **Responsive Design** with customizable inputs
- **Resource Effectiveness Analytics**
- **Professional Gauge Charts**

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

### 3. Open in Browser

The app will automatically open at `http://localhost:8501`

## 📊 How It Works

1. **Navigate** through different pages using the sidebar
2. **Enter** fire incident parameters (location, resources, containment)
3. **Predict** the severity by clicking the prediction button
4. **Analyze** results with visual gauges and severity classifications
5. **View** analytics and insights on the Analytics page

## � Demo Scenarios

The app includes **5 preset demo scenarios** for testing:

1. **Minor Fire (Small Scale)** - Well-controlled small fire
2. **Moderate Fire (Growing)** - Escalating fire situation
3. **Severe Fire (Critical)** - Large-scale emergency
4. **Contained Fire (Nearly Out)** - Fire nearing full containment
5. **Custom Input** - Manual parameter entry

Each scenario demonstrates different prediction outcomes. See `DEMO_SCENARIOS.md` for detailed information.

## �🎯 Input Parameters

- **County Code**: Encoded geographic location (0-60)
- **Latitude & Longitude**: Precise coordinates
- **Percent Contained**: Current containment percentage (0-100%)
- **Personnel Involved**: Number of firefighters deployed
- **Fire Engines**: Ground firefighting vehicles
- **Helicopters**: Aerial firefighting resources
- **Bulldozers**: Heavy equipment for firebreaks
- **Water Tenders**: Water supply vehicles
- **Major Incident**: Classification flag (Yes/No)

## 📈 Severity Levels

- ✅ **Minor Fire**: 0 - 10,000 acres (Under Control)
- ⚠️ **Moderate Fire**: 10,000 - 100,000 acres (Needs Attention)
- 🚨 **Severe Fire**: > 100,000 acres (Immediate Action Required)

## 🎨 UI Features

- **Gradient Backgrounds**: Beautiful purple-blue gradients
- **Glassmorphism Cards**: Modern frosted glass effects
- **Interactive Charts**: Plotly-powered visualizations
- **Responsive Layout**: Adapts to different screen sizes
- **Custom Styling**: Professional CSS animations
- **Emoji Integration**: Visual appeal with relevant icons

## 🛠️ Technologies

- **Streamlit**: Web application framework
- **Scikit-learn**: Machine learning models
- **Plotly**: Interactive visualizations
- **Pandas & NumPy**: Data processing
- **Joblib**: Model serialization

## 📝 Project Info

- **Course**: Computational Intelligence (CO4)
- **Institution**: MIT Academy of Engineering, Alandi (D), Pune
- **Academic Year**: 2024-25
- **Objective**: Deploy ML models for disaster management

## 🔮 Future Enhancements

- Real-time weather data integration
- Historical fire pattern analysis
- Multi-model ensemble predictions
- Mobile application
- Emergency response system integration

## 📄 License

Academic Project - MITAOE 2024-25

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please contact the development team.

---

**Made with ❤️ at MITAOE** | **Powered by Machine Learning 🤖**
