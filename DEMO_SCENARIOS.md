# 🎬 Demo Scenarios Guide

This guide provides preset scenarios to test the wildfire prediction model with different outcomes.

## 📊 Available Demo Scenarios

### 1. ✅ Minor Fire (Small Scale)
**Expected Output:** Minor Fire (0-10,000 acres)

**Scenario:** Small brush fire with good containment
- **Location:** County 15, Lat: 38.5, Long: -121.5
- **Containment:** 75% contained
- **Resources:**
  - Personnel: 25
  - Fire Engines: 5
  - Helicopters: 1
  - Bulldozers: 0
  - Water Tenders: 1
- **Major Incident:** No

**Use Case:** Testing model response to well-controlled small fires

---

### 2. ⚠️ Moderate Fire (Growing)
**Expected Output:** Moderate Fire (10,000-100,000 acres)

**Scenario:** Spreading wildfire requiring significant resources
- **Location:** County 25, Lat: 36.5, Long: -119.5
- **Containment:** 30% contained
- **Resources:**
  - Personnel: 150
  - Fire Engines: 25
  - Helicopters: 5
  - Bulldozers: 3
  - Water Tenders: 8
- **Major Incident:** Yes

**Use Case:** Testing model response to escalating fires

---

### 3. 🚨 Severe Fire (Critical)
**Expected Output:** Severe Fire (>100,000 acres)

**Scenario:** Large-scale wildfire emergency
- **Location:** County 35, Lat: 39.0, Long: -122.0
- **Containment:** 10% contained (low)
- **Resources:**
  - Personnel: 500
  - Fire Engines: 75
  - Helicopters: 15
  - Bulldozers: 10
  - Water Tenders: 20
- **Major Incident:** Yes

**Use Case:** Testing model response to critical emergency situations

---

### 4. ✅ Contained Fire (Nearly Out)
**Expected Output:** Minor Fire (0-10,000 acres)

**Scenario:** Fire almost completely contained
- **Location:** County 20, Lat: 37.8, Long: -120.8
- **Containment:** 95% contained (nearly complete)
- **Resources:**
  - Personnel: 100
  - Fire Engines: 15
  - Helicopters: 3
  - Bulldozers: 2
  - Water Tenders: 5
- **Major Incident:** No

**Use Case:** Testing model response to fires nearing full containment

---

### 5. 🔧 Custom Input
**Expected Output:** Variable (depends on your inputs)

**Scenario:** Default starting values for manual testing
- **Location:** County 10, Lat: 37.0, Long: -120.0
- **Containment:** 50% contained
- **Resources:**
  - Personnel: 50
  - Fire Engines: 10
  - Helicopters: 2
  - Bulldozers: 1
  - Water Tenders: 2
- **Major Incident:** No

**Use Case:** Base template for custom scenario testing

---

## 🎯 How to Use Demo Scenarios

1. **Navigate** to the **📊 Prediction** page
2. **Select** a demo scenario from the dropdown menu
3. **Review** the auto-filled values
4. **Adjust** any parameters if desired
5. **Click** "Predict Severity" to see results
6. **Compare** the prediction with the expected outcome

## 📊 Understanding the Results

### Severity Classifications:
- **Minor (Green):** 0 - 10,000 acres
  - Manageable with current resources
  - Standard monitoring required
  
- **Moderate (Orange):** 10,000 - 100,000 acres
  - Requires close monitoring
  - Scale up resource allocation
  
- **Severe (Red):** > 100,000 acres
  - Critical situation
  - Maximum resource deployment needed

## 💡 Tips for Testing

1. **Start with presets:** Use demo scenarios to understand model behavior
2. **Modify gradually:** Change one parameter at a time to see its impact
3. **Compare scenarios:** Switch between demos to see how inputs affect predictions
4. **Test edge cases:** Try extreme values to understand model limits
5. **Resource impact:** Notice how increasing resources affects predictions

## 🔬 Key Factors Affecting Predictions

### High Impact Factors:
- **Containment Percentage** (Higher = Lower prediction)
- **Major Incident Status** (Yes = Higher prediction)
- **Personnel Count** (More personnel = Better control)
- **Helicopter Count** (High effectiveness in containment)

### Moderate Impact Factors:
- Fire Engines
- Water Tenders
- Bulldozers
- Geographic Location

## 🎓 Educational Use

These scenarios are designed for:
- **Learning:** Understanding wildfire severity prediction
- **Training:** Practicing with the prediction model
- **Demonstration:** Showing model capabilities
- **Testing:** Validating model performance
- **Presentation:** Academic and professional showcases

---

**Note:** These demo scenarios are based on realistic wildfire incident parameters but are simplified for demonstration purposes. Actual wildfire predictions should incorporate real-time data and additional environmental factors.
