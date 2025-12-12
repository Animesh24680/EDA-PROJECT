// Configuration
const API_BASE_URL = 'http://127.0.0.1:5000/';

// DOM Elements
const form = document.getElementById('riskForm');
const resultBox = document.getElementById('resultBox');
const resultIcon = document.getElementById('resultIcon');
const resultText = document.getElementById('resultText');
const resultMessage = document.getElementById('resultMessage');

// Form Submission Handler
form.addEventListener('submit', function(event) {
    event.preventDefault();
    
    // Gather form data
    const age = parseFloat(document.getElementById('age').value);
    const income = parseFloat(document.getElementById('income').value);
    const creditScore = parseFloat(document.getElementById('creditScore').value);
    const missedPayments = parseInt(document.getElementById('missedPayments').value);
    const debtToIncomeRatio = parseFloat(document.getElementById('debtToIncomeRatio').value);
    
    // Validate inputs
    if (!validateInputs(age, income, creditScore, missedPayments, debtToIncomeRatio)) {
        alert('Please enter valid input values.');
        return;
    }
    
    // Send prediction request
    predictDelinquency(age, income, creditScore, missedPayments, debtToIncomeRatio);
});

// Prediction Function
async function predictDelinquency(age, income, creditScore, missedPayments, debtToIncomeRatio) {
    try {
        const response = await fetch(`${API_BASE_URL}/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                age: age,
                income: income,
                creditScore: creditScore,
                missedPayments: missedPayments,
                debtToIncomeRatio: debtToIncomeRatio
            })
        });
        
        if (!response.ok) {
            throw new Error('Prediction request failed');
        }
        
        const result = await response.json();
        updateResultDisplay(result);
        
    } catch (error) {
        console.error('Prediction Error:', error);
        fallbackPrediction(creditScore, missedPayments, debtToIncomeRatio);
    }
}

// Fallback Prediction Method
function fallbackPrediction(creditScore, missedPayments, debtToIncomeRatio) {
    const riskFactors = [
        creditScore < 600,
        missedPayments > 2,
        debtToIncomeRatio > 0.4
    ].filter(Boolean).length;
    
    const result = {
        is_delinquent: riskFactors >= 2,
        risk_probability: riskFactors / 3,
        message: riskFactors >= 2 
            ? "High risk of financial delinquency detected." 
            : "Low risk of financial delinquency.",
        method: 'local_fallback'
    };
    
    updateResultDisplay(result);
}

// Update Result Display
function updateResultDisplay(result) {
    // Show the result box
    resultBox.classList.add('visible');
    
    // Use setTimeout to trigger the fade-in animation
    setTimeout(() => {
        resultBox.classList.add('show');
    }, 10);
    
    resultBox.classList.remove('high-risk', 'low-risk');
    
    if (result.is_delinquent) {
        resultBox.classList.add('high-risk');
        resultIcon.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                <line x1="12" y1="9" x2="12" y2="13"/>
                <line x1="12" y1="17" x2="12.01" y2="17"/>
            </svg>
        `;
        resultText.textContent = 'High Risk';
    } else {
        resultBox.classList.add('low-risk');
        resultIcon.innerHTML = `
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M22 11.08V12a10 10 0 1 1-5.93-8.64"/>
                <path d="M22 4L12 14.01l-3-3"/>
            </svg>
        `;
        resultText.textContent = 'Low Risk';
    }
    
    // Risk Probability Formatting
    const riskPercentage = (result.risk_probability * 100).toFixed(1);
    resultMessage.textContent = `${result.message} Risk Probability: ${riskPercentage}%`;
}

// Input Validation
function validateInputs(age, income, creditScore, missedPayments, debtToIncomeRatio) {
    return (
        age > 0 && age < 120 &&
        income > 0 &&
        creditScore > 300 && creditScore <= 850 &&
        missedPayments >= 0 &&
        debtToIncomeRatio >= 0 && debtToIncomeRatio <= 1
    );
}

// Optional: Initial Model Training
async function trainModel() {
    try {
        const response = await fetch(`${API_BASE_URL}/train`, {
            method: 'POST'
        });
        const result = await response.json();
        console.log('Model Training Result:', result);
    } catch (error) {
        console.error('Model Training Error:', error);
    }
}

// Train model on page load (optional)
document.addEventListener('DOMContentLoaded', trainModel);