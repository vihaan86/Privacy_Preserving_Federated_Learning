const API_URL = 'http://localhost:5000';

// Navigation
document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', (e) => {
        e.preventDefault();
        
        // Remove active class
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
        
        // Add active class
        link.classList.add('active');
        const pageId = link.getAttribute('data-page');
        document.getElementById(pageId).classList.add('active');
    });
});

// Form submission
document.getElementById('predictionForm').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btn = document.querySelector('.btn-predict');
    btn.disabled = true;
    btn.innerHTML = '⏳ Analyzing...';
    
    try {
        // Get form data
        const formData = new FormData(document.getElementById('predictionForm'));
        const data = Object.fromEntries(formData);
        
        // Convert to numbers
        const payload = {
            age: parseFloat(data.age),
            sex: parseInt(data.sex),
            cp: parseInt(data.cp),
            trestbps: parseFloat(data.trestbps),
            chol: parseFloat(data.chol),
            fbs: parseInt(data.fbs),
            restecg: parseInt(data.restecg),
            thalach: parseFloat(data.thalach),
            exang: parseInt(data.exang),
            oldpeak: parseFloat(data.oldpeak),
            slope: parseInt(data.slope),
            ca: parseInt(data.ca),
            thal: parseInt(data.thal)
        };
        
        console.log('Sending payload:', payload);
        
        // Call backend
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload)
        });
        
        const result = await response.json();
        
        if (result.success) {
            displayResult(result);
        } else {
            alert('Error: ' + result.error);
        }
    } catch (error) {
        console.error('Error:', error);
        alert('Failed to connect to backend. Make sure Flask is running on localhost:5000');
    } finally {
        btn.disabled = false;
        btn.innerHTML = '🔍 Predict Heart Disease Risk';
    }
});

function displayResult(result) {
    const resultSection = document.getElementById('resultSection');
    const predictionResult = document.getElementById('predictionResult');
    const riskPercentage = document.getElementById('riskPercentage');
    const confidence = document.getElementById('confidence');
    const recommendation = document.getElementById('recommendation');
    
    const isPredictionHigh = result.prediction > 0.5;
    const riskClass = isPredictionHigh ? 'high-risk' : 'low-risk';
    
    // Prediction box
    predictionResult.className = `prediction-box ${riskClass}`;
    if (isPredictionHigh) {
        predictionResult.innerHTML = `
            ⚠️ HIGH RISK DETECTED<br>
            <span class="risk-percent">${result.risk_percentage}%</span>
        `;
    } else {
        predictionResult.innerHTML = `
            ✅ LOW RISK<br>
            <span class="risk-percent">${result.risk_percentage}%</span>
        `;
    }
    
    // Metrics
    riskPercentage.textContent = result.risk_percentage + '%';
    confidence.textContent = result.confidence.toFixed(1) + '%';
    
    // Recommendation
    recommendation.className = `recommendation ${isPredictionHigh ? 'danger' : 'success'}`;
    if (isPredictionHigh) {
        recommendation.innerHTML = `
            <h4>🚨 Recommendation</h4>
            <p>Please consult a cardiologist immediately for comprehensive evaluation and treatment planning.</p>
        `;
    } else {
        recommendation.innerHTML = `
            <h4>✅ Recommendation</h4>
            <p>Good news! Current indicators suggest low risk. Maintain a healthy lifestyle with regular exercise and balanced diet. Schedule annual checkups.</p>
        `;
    }
    
    // Show result section
    resultSection.style.display = 'block';
    resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// Check backend on page load
window.addEventListener('load', async () => {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend connected');
        }
    } catch (error) {
        console.warn('⚠️ Backend not available:', error);
    }
});
