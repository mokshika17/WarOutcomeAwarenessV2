from flask import Flask, render_template, send_file, request, jsonify, url_for, session
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import io
import base64
import joblib
import uuid
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor
import xgboost as xgb

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-here-change-in-production")

# Configure Gemini AI
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Nuclear Chatbot Class
class NuclearConflictChatbot:
    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.system_prompt = """
        You are an expert geopolitical analyst specializing in India-Pakistan nuclear relations and conflict scenarios. 
        
        Your role is to provide:
        - Educational analysis of nuclear doctrines and policies
        - Historical context of India-Pakistan conflicts
        - Risk assessment and escalation scenarios
        - Diplomatic and peace-building perspectives
        - Strategic stability concepts
        - Arms control and confidence-building measures
        
        Guidelines:
        - Maintain academic objectivity and neutrality
        - Focus on factual analysis rather than speculation
        - Emphasize peace-building and de-escalation when possible
        - Provide historical context for current tensions
        - Discuss both countries' perspectives fairly
        - Include relevant dates, treaties, and key events
        - Avoid sensationalism or fear-mongering
        - Encourage diplomatic solutions
        - Keep responses concise but informative (2-3 paragraphs max)
        
        Always frame responses in terms of conflict prevention, strategic stability, and regional peace.
        """
        
    def get_response(self, question, chat_history=None):
        try:
            # Construct the full prompt with context
            full_prompt = f"""
            {self.system_prompt}
            
            Previous conversation context: {chat_history if chat_history else 'None'}
            
            Current question: {question}
            
            Please provide a comprehensive, balanced analysis focusing on factual information and peace-building perspectives.
            Keep your response concise but informative.
            """
            
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            return f"I apologize, but I encountered an error: {str(e)}. Please try rephrasing your question."

# Initialize chatbot
chatbot = NuclearConflictChatbot()

# Step 1: Data Simulation (Synthetic) - Your existing code
def generate_simulation():
    np.random.seed(42)
    n_simulations = 10000

    data = {
        "nukes_used": np.random.randint(1, 50, size=n_simulations),
        "avg_yield_kt": np.random.uniform(15, 1000, size=n_simulations),
        "target_city_population": np.random.uniform(1e5, 2e7, size=n_simulations),
        "urbanization_level": np.random.uniform(0.4, 1.0, size=n_simulations),
        "soot_emission_Tg": np.random.uniform(0.1, 10, size=n_simulations),
    }
    
    data["total_targeted_population"] = data["target_city_population"] * data["nukes_used"] * data["urbanization_level"]
    data["human_lives_lost_millions"] = data["total_targeted_population"] * np.random.uniform(0.2, 0.8, size=n_simulations) / 1e6
    data["gdp_impact_pct"] = np.clip(data["nukes_used"] * np.random.uniform(0.5, 1.5, size=n_simulations), 5, 100)
    data["expected_global_temp_drop_C"] = data["soot_emission_Tg"] * np.random.uniform(0.05, 0.2, size=n_simulations)
    data["estimated_famine_risk_millions"] = data["expected_global_temp_drop_C"] * np.random.uniform(50, 500, size=n_simulations)

    return pd.DataFrame(data)

# Global variables - Your existing code
df_sim = generate_simulation()
features = ['nukes_used', 'avg_yield_kt', 'target_city_population', 'urbanization_level', 'soot_emission_Tg']
targets = ['human_lives_lost_millions', 'gdp_impact_pct', 'expected_global_temp_drop_C', 'estimated_famine_risk_millions']

# Initialize global variables for best model and scaler
best_model = None
best_model_name = ""
scaler = StandardScaler()

# Your existing ML functions (keeping them unchanged)
def train_and_compare_models():
    """Train both RandomForest and XGBoost models, compare performance, and select the best one."""
    global best_model, best_model_name, scaler
    
    print("Training and comparing models...")
    
    # Prepare data
    X = df_sim[features]
    y = df_sim[targets]
    
    # Scale features
    X_scaled = scaler.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    # Initialize models
    models = {
        'RandomForest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)),
        'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    }
    
    # Train and evaluate models
    model_scores = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics for each target
        target_scores = {}
        for i, target in enumerate(targets):
            mse = mean_squared_error(y_test.iloc[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test.iloc[:, i], y_pred[:, i])
            r2 = r2_score(y_test.iloc[:, i], y_pred[:, i])
            
            target_scores[target] = {
                'mse': mse,
                'mae': mae,
                'r2': r2
            }
        
        # Calculate overall score (average R2 across all targets)
        overall_r2 = np.mean([scores['r2'] for scores in target_scores.values()])
        overall_mae = np.mean([scores['mae'] for scores in target_scores.values()])
        
        model_scores[name] = {
            'overall_r2': overall_r2,
            'overall_mae': overall_mae,
            'target_scores': target_scores
        }
        
        print(f"{name} - Overall R2: {overall_r2:.4f}, Overall MAE: {overall_mae:.4f}")
    
    # Select best model based on R2 score
    best_model_name = max(model_scores.keys(), key=lambda k: model_scores[k]['overall_r2'])
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model: {best_model_name}")
    print(f"Best R2 Score: {model_scores[best_model_name]['overall_r2']:.4f}")
    print(f"Best MAE Score: {model_scores[best_model_name]['overall_mae']:.4f}")
    
    # Print detailed scores for best model
    print(f"\nDetailed scores for {best_model_name}:")
    for target, scores in model_scores[best_model_name]['target_scores'].items():
        print(f"  {target}:")
        print(f"    R2: {scores['r2']:.4f}")
        print(f"    MAE: {scores['mae']:.4f}")
        print(f"    MSE: {scores['mse']:.4f}")
    
    # Save the best model and scaler
    joblib.dump(best_model, 'best_nuclear_model.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    joblib.dump(best_model_name, 'best_model_name.joblib')
    
    print(f"\nBest model ({best_model_name}) and scaler saved successfully!")
    
    return model_scores

def load_best_model():
    """Load the best model and scaler from disk."""
    global best_model, best_model_name, scaler
    
    try:
        best_model = joblib.load('best_nuclear_model.joblib')
        scaler = joblib.load('feature_scaler.joblib')
        best_model_name = joblib.load('best_model_name.joblib')
        print(f"Loaded best model: {best_model_name}")
        return True
    except FileNotFoundError:
        print("No saved model found. Training new models...")
        return False

def create_visualizations():
    if not os.path.exists('static'):
        os.makedirs('static')
        
    sns.set(style="whitegrid")

    # Pairplot
    pairplot_path = 'static/pairplot.png'
    sns.pairplot(df_sim[["nukes_used", "avg_yield_kt", "human_lives_lost_millions", "gdp_impact_pct"]])
    plt.savefig(pairplot_path)
    plt.clf()

    # Soot vs Temperature Drop
    soot_temp_path = 'static/soot_temp.png'
    plt.figure(figsize=(10, 6))
    sns.regplot(x='soot_emission_Tg', y='expected_global_temp_drop_C', data=df_sim)
    plt.title('Soot Emission vs. Global Temperature Drop')
    plt.xlabel('Soot Emission (Tg)')
    plt.ylabel('Expected Global Temperature Drop (°C)')
    plt.savefig(soot_temp_path)
    plt.clf()

    # Histogram plots
    dist_path = 'static/distribution.png'
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(df_sim['human_lives_lost_millions'], kde=True)
    plt.title('Distribution of Human Lives Lost (Millions)')
    plt.subplot(1, 2, 2)
    sns.histplot(df_sim['gdp_impact_pct'], kde=True)
    plt.title('Distribution of GDP Impact (%)')
    plt.tight_layout()
    plt.savefig(dist_path)
    plt.clf()

    # Boxplot
    boxplot_path = 'static/boxplot.png'
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='nukes_used', y='gdp_impact_pct', data=df_sim)
    plt.title('GDP Impact vs. Number of Nukes Used')
    plt.savefig(boxplot_path)
    plt.clf()

# Your existing routes
@app.route('/download')
def download():
    csv_path = 'nuclear_winter_simulations.csv'
    df_sim.to_csv(csv_path, index=False)
    return send_file(csv_path, as_attachment=True)

@app.route('/')
def home():
    # Initialize chat session if not exists
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
        session['chat_history'] = []
    
    return render_template('index.html', 
                         best_model=best_model_name if best_model_name else "Unknown")

@app.route('/model_info')
def model_info():
    """Route to display model comparison information."""
    if best_model_name:
        return jsonify({
            'best_model': best_model_name,
            'status': 'Model loaded successfully'
        })
    else:
        return jsonify({
            'best_model': 'No model loaded',
            'status': 'No model available'
        })

@app.route('/retrain', methods=['POST'])
def retrain_models():
    """Route to retrain and compare models."""
    try:
        model_scores = train_and_compare_models()
        return jsonify({
            'status': 'success',
            'best_model': best_model_name,
            'scores': model_scores
        })
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    # Initialize chat session if not exists
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
        session['chat_history'] = []
        
    if request.method == 'POST':
        try:
            print("Processing POST request to /predict")
            
            # Check if model is loaded
            if best_model is None:
                raise Exception("No model loaded. Please retrain models first.")
            
            # Get form values
            nukes_used = int(request.form.get('nukes_used'))
            avg_yield_kt = float(request.form.get('avg_yield_kt'))
            target_population = int(request.form.get('target_city_population'))
            urbanization = float(request.form.get('urbanization_level'))
            soot_emission = float(request.form.get('soot_emission_Tg'))
            
            print(f"Input values: nukes={nukes_used}, yield={avg_yield_kt}, pop={target_population}, urban={urbanization}, soot={soot_emission}")
            
            # Create input feature array
            input_features = np.array([[nukes_used, avg_yield_kt, target_population, 
                                    urbanization, soot_emission]])
            
            # Scale input features
            scaled_input = scaler.transform(input_features)
            
            # Make prediction using the best model
            prediction_array = best_model.predict(scaled_input)[0]
            
            # Convert to dictionary
            predictions = {}
            for i, target in enumerate(targets):
                predictions[target] = float(prediction_array[i])
            
            print(f"Generated predictions using {best_model_name}: {predictions}")

            # === Enhanced Visualization Section ===
            print("Generating visualization...")
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import io
            import base64
            try:                
                # Create visualization using BytesIO to send the image directly
                plt.figure(figsize=(10, 6))
                categories = list(predictions.keys())
                values = list(predictions.values())
                
                # Make more readable labels
                display_categories = [
                    'Lives Lost (M)', 
                    'GDP Impact (%)', 
                    'Temp Drop (°C)', 
                    'Famine Risk (M)'
                ]
                
                plt.bar(display_categories, values, color='skyblue')
                plt.title('Nuclear Winter Impact Predictions')
                plt.ylabel('Estimated Value')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                
                # Save the figure to a BytesIO object
                img_io = io.BytesIO()
                plt.savefig(img_io, format='png')
                img_io.seek(0)
                plt.close()
                
                # Convert to base64 for embedding directly in HTML
                img_data = base64.b64encode(img_io.getvalue()).decode('utf-8')
                img_src = f"data:image/png;base64,{img_data}"
                
                print("Image generated and encoded successfully")
                print(f"Image data starts with: {img_src[:50]}...")

            except Exception as viz_error:
                print(f"Visualization error: {viz_error}")
                import traceback
                print(traceback.format_exc())
                img_src = None

                print("Image generated and encoded successfully")
                print(f"Image data length: {len(img_data)} characters")
                
            return render_template('predict.html', 
                               predictions=predictions, 
                               img_src=img_src,
                               model_name=best_model_name)

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            print(f"Error in predict route: {error_details}")
            return render_template('predict.html', error=str(e))

    return render_template('predict.html', model_name=best_model_name)

@app.route('/chatbot', methods=['GET'])
def chatbot_page():
    """Render the chatbot interface"""
    # Initialize chat session if not exists
    if 'chat_id' not in session:
        session['chat_id'] = str(uuid.uuid4())
        session['chat_history'] = []
    
    return render_template('chatbot.html', chat_id=session['chat_id'])

# CHATBOT ROUTES
@app.route('/chatbot/chat', methods=['POST'])
def chatbot_chat():
    """Handle chatbot messages"""
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        
        if not user_message:
            return jsonify({'error': 'Please enter a valid question'}), 400
        
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Format chat history for context (last 5 exchanges)
        context = []
        for item in chat_history[-10:]:  # Last 5 exchanges (10 items = 5 Q&A pairs)
            context.append(f"Q: {item['message']}" if item['type'] == 'user' else f"A: {item['message']}")
        
        context_string = "\n".join(context) if context else None
        
        # Get response from chatbot
        bot_response = chatbot.get_response(user_message, context_string)
        
        # Update chat history
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        chat_history.append({
            'type': 'user',
            'message': user_message,
            'timestamp': timestamp
        })
        
        chat_history.append({
            'type': 'bot',
            'message': bot_response,
            'timestamp': timestamp
        })
        
        # Keep only last 20 messages to prevent session from getting too large
        if len(chat_history) > 20:
            chat_history = chat_history[-20:]
        
        session['chat_history'] = chat_history[-20:]  # Store only the last 20 messages
        
        return jsonify({
            'response': bot_response,
            'timestamp': timestamp
        })
        
    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/chatbot/clear', methods=['POST'])
def clear_chatbot():
    """Clear chat history"""
    session['chat_history'] = []
    return jsonify({'message': 'Chat history cleared successfully'})

@app.route('/chatbot/history')
def get_chatbot_history():
    """Get chat history"""
    return jsonify({'history': session.get('chat_history', [])})

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    return render_template('error.html', error="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', error="Internal server error"), 500

if __name__ == '__main__':
    # Check for required environment variables
    if not os.getenv("GOOGLE_API_KEY"):
        print("Warning: GOOGLE_API_KEY not found in environment variables")
        print("Chatbot functionality will not work without API key")
        print("Please add GOOGLE_API_KEY to your .env file")
    
    # Try to load existing model, if not available, train new ones
    if not load_best_model():
        train_and_compare_models()
    
    # Create static directory and visualizations
    create_visualizations()
    
    print(f"Starting Flask app with {best_model_name} model...")
    print("Chatbot popup will be available on all pages")
    app.run(debug=True)