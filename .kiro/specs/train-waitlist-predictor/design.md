# Train Waitlist Predictor - Design Document

## Overview

The Train Waitlist Predictor is a web-based application that uses machine learning to predict the confirmation probability of waitlisted train tickets. The system consists of a frontend interface for user input, a backend API for processing requests, a machine learning model for predictions, and a data pipeline for collecting and processing historical booking data.

The application follows a client-server architecture with a Python-based backend (Flask/FastAPI) and a modern web frontend (React or vanilla JavaScript). The prediction engine uses a trained ML model (Random Forest or Gradient Boosting) that considers multiple features including waitlist position, days to journey, historical confirmation rates, class type, quota, and seasonal patterns.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Web Frontend  │
│  (React/HTML)   │
└────────┬────────┘
         │ HTTP/REST
         ▼
┌─────────────────┐
│   API Server    │
│  (Flask/FastAPI)│
└────────┬────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌─────────┐ ┌──────────────┐
│Database │ │  ML Model    │
│(SQLite/ │ │  (Scikit-    │
│PostgreSQL)│ │   learn)     │
└─────────┘ └──────────────┘
```

### Component Layers

1. **Presentation Layer**: Web-based UI for user interaction
2. **API Layer**: RESTful endpoints for handling requests
3. **Business Logic Layer**: Prediction engine and data processing
4. **Data Layer**: Database for historical data and user queries
5. **ML Layer**: Trained model and feature engineering pipeline

## Components and Interfaces

### 1. Frontend Component

**Responsibilities:**
- Render input form for ticket details
- Validate user input client-side
- Display prediction results with visualizations
- Handle batch prediction requests
- Manage saved predictions

**Key Interfaces:**
- Input form with fields: train number, date, class, waitlist position, quota
- Results display with percentage, confidence level, and explanation
- Comparison view for multiple predictions
- Error message display

**Technology:** React (or vanilla JavaScript with HTML/CSS)

### 2. API Server Component

**Responsibilities:**
- Expose REST endpoints for predictions
- Validate incoming requests
- Route requests to prediction engine
- Handle authentication (if needed)
- Manage rate limiting

**Key Endpoints:**

```
POST /api/predict
Request Body:
{
  "train_number": "12345",
  "journey_date": "2025-12-15",
  "class": "3A",
  "waitlist_position": 15,
  "quota": "GN"
}

Response:
{
  "confirmation_probability": 75.5,
  "confidence": "high",
  "factors": {
    "waitlist_position_impact": "positive",
    "days_to_journey": 45,
    "historical_rate": 0.82
  },
  "explanation": "Based on historical data, this train typically confirms up to WL 20 for this class and date range."
}

POST /api/predict/batch
Request Body:
{
  "predictions": [
    { "train_number": "12345", "journey_date": "2025-12-15", ... },
    { "train_number": "12345", "journey_date": "2025-12-16", ... }
  ]
}

GET /api/train/{train_number}/stats
Response: Historical confirmation statistics for a train
```

**Technology:** Flask or FastAPI (Python)

### 3. Prediction Engine Component

**Responsibilities:**
- Load and prepare input features
- Execute ML model inference
- Calculate confidence levels
- Generate explanations
- Handle edge cases (insufficient data)

**Key Functions:**

```python
def predict_confirmation(ticket_details: dict) -> dict:
    """
    Main prediction function
    Returns: {probability, confidence, factors, explanation}
    """
    
def prepare_features(ticket_details: dict) -> np.array:
    """
    Feature engineering: extract and transform input features
    """
    
def calculate_confidence(prediction: float, feature_variance: float) -> str:
    """
    Determine confidence level based on model certainty
    """
    
def generate_explanation(features: dict, prediction: float) -> str:
    """
    Create human-readable explanation of prediction
    """
```

**ML Model Features:**
- Waitlist position (normalized)
- Days until journey
- Class type (encoded)
- Quota type (encoded)
- Train route popularity score
- Historical confirmation rate for train/class
- Day of week
- Month/season indicator
- Booking trend (recent confirmation patterns)

**Technology:** Scikit-learn (Random Forest or Gradient Boosting), Pandas, NumPy

### 4. Data Management Component

**Responsibilities:**
- Store historical booking data
- Cache prediction results
- Manage train metadata
- Track model performance metrics

**Database Schema:**

```sql
-- Historical booking data
CREATE TABLE bookings (
    id INTEGER PRIMARY KEY,
    train_number VARCHAR(10),
    journey_date DATE,
    class VARCHAR(5),
    quota VARCHAR(5),
    initial_waitlist INTEGER,
    final_status VARCHAR(20),
    days_before_journey INTEGER,
    confirmed BOOLEAN,
    created_at TIMESTAMP
);

-- Train metadata
CREATE TABLE trains (
    train_number VARCHAR(10) PRIMARY KEY,
    train_name VARCHAR(100),
    route VARCHAR(200),
    popularity_score FLOAT
);

-- Prediction cache
CREATE TABLE prediction_cache (
    id INTEGER PRIMARY KEY,
    train_number VARCHAR(10),
    journey_date DATE,
    class VARCHAR(5),
    waitlist_position INTEGER,
    prediction FLOAT,
    created_at TIMESTAMP,
    expires_at TIMESTAMP
);

-- Model performance tracking
CREATE TABLE model_metrics (
    id INTEGER PRIMARY KEY,
    model_version VARCHAR(20),
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT,
    evaluated_at TIMESTAMP
);
```

**Technology:** SQLite (development) or PostgreSQL (production)

### 5. Model Training Pipeline Component

**Responsibilities:**
- Collect and preprocess training data
- Train and validate ML models
- Evaluate model performance
- Deploy updated models
- Schedule periodic retraining

**Training Process:**

1. **Data Collection**: Aggregate historical booking data from database
2. **Feature Engineering**: Create derived features (confirmation rates, trends)
3. **Data Splitting**: Train/validation/test split (70/15/15)
4. **Model Training**: Train Random Forest or Gradient Boosting classifier
5. **Hyperparameter Tuning**: Grid search or random search for optimal parameters
6. **Validation**: Evaluate on validation set, check for overfitting
7. **Model Serialization**: Save trained model using joblib or pickle
8. **Deployment**: Replace production model if performance improves

**Technology:** Scikit-learn, Pandas, Joblib

## Data Models

### Input Data Model

```python
class PredictionRequest:
    train_number: str          # e.g., "12345"
    journey_date: date         # Future date within 120 days
    class_type: str            # "1A", "2A", "3A", "SL", etc.
    waitlist_position: int     # 1-500
    quota: str                 # "GN", "TQ", "PT", etc.
```

### Output Data Model

```python
class PredictionResponse:
    confirmation_probability: float  # 0.0 - 100.0
    confidence: str                  # "high", "medium", "low"
    factors: dict                    # Key factors influencing prediction
    explanation: str                 # Human-readable explanation
    timestamp: datetime              # When prediction was made
```

### Feature Vector Model

```python
# Features used by ML model
features = [
    'waitlist_position_normalized',    # 0-1 scale
    'days_to_journey',                 # Integer
    'class_encoded',                   # One-hot encoded
    'quota_encoded',                   # One-hot encoded
    'historical_confirmation_rate',    # 0-1 for this train/class
    'route_popularity',                # 0-1 score
    'day_of_week',                     # 0-6
    'month',                           # 1-12
    'is_holiday_season',               # Boolean
    'recent_trend'                     # Moving average of confirmations
]
```

## Error Handling

### Error Categories and Responses

1. **Validation Errors** (400 Bad Request)
   - Invalid train number format
   - Past or too-distant journey date
   - Invalid waitlist position
   - Missing required fields
   - Response: Clear error message indicating which field is invalid

2. **Data Unavailability** (503 Service Unavailable)
   - Database connection failure
   - Historical data not available for train
   - Response: Inform user, offer fallback prediction with lower confidence

3. **Model Errors** (500 Internal Server Error)
   - Model file not found
   - Prediction computation failure
   - Response: Generic error message, log details for debugging

4. **Timeout Errors** (504 Gateway Timeout)
   - Prediction takes longer than 15 seconds
   - Response: Inform user, offer retry option

### Error Handling Strategy

```python
try:
    # Validate input
    validate_request(request_data)
    
    # Prepare features
    features = prepare_features(request_data)
    
    # Get prediction
    prediction = model.predict(features)
    
    # Generate response
    response = format_response(prediction, features)
    
except ValidationError as e:
    return {"error": str(e), "code": "VALIDATION_ERROR"}, 400
    
except DataUnavailableError as e:
    # Use fallback prediction
    response = fallback_prediction(request_data)
    response["warning"] = "Limited data available, prediction may be less accurate"
    return response, 200
    
except ModelError as e:
    log_error(e)
    return {"error": "Unable to generate prediction", "code": "MODEL_ERROR"}, 500
    
except TimeoutError as e:
    return {"error": "Request timeout", "code": "TIMEOUT", "retry": True}, 504
```

## Testing Strategy

### 1. Unit Tests

**Frontend:**
- Input validation logic
- Form submission handling
- Result rendering components
- Error message display

**Backend:**
- API endpoint request/response handling
- Feature engineering functions
- Prediction engine logic
- Database query functions

**ML Model:**
- Feature preparation correctness
- Model prediction output format
- Confidence calculation logic

### 2. Integration Tests

- End-to-end API request flow
- Database read/write operations
- Model loading and inference
- Cache functionality
- Batch prediction processing

### 3. Model Validation Tests

- Prediction accuracy on test dataset (target: >70%)
- Precision and recall metrics
- Performance on edge cases (WL 1, WL 500, same-day journey)
- Consistency across similar inputs
- Prediction time performance (<5 seconds)

### 4. User Acceptance Testing

- Manual testing of UI workflows
- Prediction result clarity and usefulness
- Error message comprehensibility
- Performance under normal load

### 5. Performance Tests

- API response time under load
- Concurrent request handling
- Database query optimization
- Model inference speed
- Memory usage monitoring

### Test Data Strategy

- Use synthetic historical booking data for development
- Create test fixtures for various scenarios:
  - High confirmation probability cases
  - Low confirmation probability cases
  - Edge cases (extreme waitlist positions)
  - Missing data scenarios
- Mock external data sources for unit tests
- Use separate test database for integration tests

## Deployment Considerations

### Initial MVP Scope

For the first version, focus on:
- Single prediction endpoint (not batch)
- Basic web interface with essential fields
- Pre-trained model with synthetic data
- SQLite database
- Simple deployment (single server)

### Future Enhancements

- Real-time data integration with railway APIs
- User accounts and prediction history
- Mobile app version
- Advanced analytics and insights
- A/B testing for model improvements
- Notification system for confirmation updates
