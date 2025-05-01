"""
Main Flask server for Quran reciter and ayah identification API endpoints.
"""
import logging
from flask import Flask, request, jsonify
import numpy as np
import json
from pathlib import Path

from src.utils.audio_utils import process_audio_file
from src.features import extract_features
from server.utils.model_loader import initialize_models, get_reciter_model, get_ayah_model
from src.utils.ayah_matching import find_matching_ayah
from src.utils.distance_utils import calculate_distances, analyze_prediction_reliability
from server.config import TOP_N_PREDICTIONS

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Initialize models at startup
reciter_success, ayah_success = initialize_models()
if not (reciter_success and ayah_success):
    logger.error("Failed to initialize one or more models")
    # In production, you might want to exit here
    # import sys; sys.exit(1)

@app.route('/getAyah', methods=['POST'])
def get_ayah():
    """
    Identify a Quranic verse from an audio recording.
    
    Accepts:
        - audio: Audio file (MP3/WAV, 1-10 seconds)
        - max_matches: Max results to return (optional)
        - min_confidence: Confidence threshold (optional)
    """
    try:
        # Validate request has audio file
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided. Please send the file with key "audio" in form data.'
            }), 400

        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({
                'error': 'No audio file selected'
            }), 400

        # Parse optional parameters
        max_matches = int(request.form.get('max_matches', TOP_N_PREDICTIONS))
        min_confidence = float(request.form.get('min_confidence', 0.70))
        
        # Process and validate audio
        result = process_audio_file(audio_file, for_ayah=True)
        if isinstance(result[1], str):
            return jsonify({'error': result[1]}), 400

        audio_data, sample_rate = result
        
        # Generate transcription
        model = get_ayah_model()
        transcribed_text = model.transcribe(audio_data, sample_rate)
        if not transcribed_text:
            return jsonify({'error': 'Failed to transcribe audio'}), 500
            
        # Find matching verses
        matches = find_matching_ayah(
            transcribed_text,
            min_confidence=min_confidence,
            max_matches=max_matches
        )
        
        matches_list = matches.get('matches', [])
        
        # Build response
        response = {
            'matches_found': len(matches_list) > 0,
            'total_matches': len(matches_list),
            'matches': matches_list,
            'best_match': matches_list[0] if matches_list else None
        }
        
        # Add debug info in development
        if app.debug:
            debug_response = {
                'transcription': transcribed_text,
                'debug_info': {
                    'transcription': transcribed_text,
                    'normalized_transcription': matches.get('normalized_input', ''),
                    'normalized_matches': matches.get('normalized_matches', [])
                }
            }
            response.update(debug_response)
        
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error in transcription or matching: {str(e)}")
        return jsonify({
            'error': f'Error in transcription or matching: {str(e)}'
        }), 500

@app.route('/getReciter', methods=['POST'])
def identify_reciter():
    """
    Identify a Quran reciter from an audio recording.
    
    Accepts:
        - audio: Audio file (MP3/WAV, 5-15 seconds)
        - show_unreliable: Show predictions below confidence threshold (optional)
    """
    try:
        # Validate request has audio file
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided. Please send the file with key "audio" in form data.'
            }), 400

        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({
                'error': 'No audio file selected'
            }), 400

        show_unreliable = request.form.get('show_unreliable_predictions', '').lower() == 'true'
        
        # Process and validate audio
        result = process_audio_file(audio_file, for_ayah=False)
        if isinstance(result[1], str):
            return jsonify({'error': result[1]}), 400

        audio_data, sample_rate = result
        
        # Load model and validate
        model = get_reciter_model()
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
        # Extract audio features
        try:
            features = extract_features(audio_data, sample_rate)
            if features is None:
                return jsonify({'error': 'Feature extraction failed'}), 500
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return jsonify({
                'error': f'Error extracting features: {str(e)}'
            }), 500

        try:
            # Reshape features if needed
            if features.ndim == 1:
                features = features.reshape(1, -1)

            # Get model predictions
            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            # Calculate reliability metrics
            distances = calculate_distances(features[0], model.centroids)
            reliability = analyze_prediction_reliability(
                probabilities, distances, model.thresholds, prediction)

            # Get top predictions
            sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
            predictions = []
            
            # Load reciter metadata
            reciters_data = {}
            try:
                reciters_file = Path(__file__).resolve().parent.parent / 'data' / 'reciters.json'
                if reciters_file.exists():
                    with open(reciters_file, 'r', encoding='utf-8') as f:
                        reciters_data = json.load(f)
                else:
                    logger.warning(f"Reciters metadata file not found at: {reciters_file}")
            except Exception as e:
                logger.warning(f"Could not load reciters metadata: {str(e)}")
            
            for idx in sorted_indices:
                reciter_name = str(model.classes_[idx])
                confidence = float(probabilities[idx])
                
                # Get reciter metadata if available
                reciter_info = reciters_data.get(reciter_name, {})
                servers = reciter_info.get('servers', [])
                server_url = servers[0] if isinstance(servers, list) and servers else ''
                
                predictions.append({
                    'name': reciter_name,
                    'confidence': confidence * 100,  # Convert to percentage
                    'nationality': reciter_info.get('nationality', ''),
                    'serverUrl': server_url,
                    'flagUrl': reciter_info.get('flagUrl', ''),
                    'imageUrl': reciter_info.get('imageUrl', '')
                })

            # Format response based on reliability
            if not reliability['is_reliable'] and not show_unreliable:
                response = {
                    'reliable': False,
                    'main_prediction': None,
                    'top_predictions': []
                }
            else:
                response = {
                    'reliable': bool(reliability['is_reliable']),
                    'main_prediction': predictions[0] if predictions else None,
                    'top_predictions': predictions
                }

            return jsonify(response)

        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            return jsonify({
                'error': f'Error making prediction: {str(e)}'
            }), 500

    except Exception as e:
        logger.error(f"Server error: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

@app.route('/getAllReciters', methods=['GET'])
def get_all_reciters():
    """Endpoint to get a list of all available reciters."""
    try:
        # Load reciters data
        try:
            reciters_file = Path(__file__).resolve().parent.parent / 'data' / 'reciters.json'
            if not reciters_file.exists():
                return jsonify({'error': 'Reciters data file not found.'}), 500
                
            with open(reciters_file, 'r', encoding='utf-8') as f:
                reciters_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading reciters data: {str(e)}")
            return jsonify({'error': 'Error loading reciters data.'}), 500

        # Format reciters list
        reciters_list = []
        for name, data in reciters_data.items():
            servers = data.get('servers', [])
            server_url = servers[0] if isinstance(servers, list) and servers else ''
            if isinstance(servers, str):
                server_url = servers
                
            reciter_info = {
                'name': name,
                'nationality': data.get('nationality', 'Unknown'),
                'flagUrl': data.get('flagUrl', ''),
                'imageUrl': data.get('imageUrl', ''),
                'serverUrl': server_url
            }
            reciters_list.append(reciter_info)
            
        return jsonify(reciters_list), 200
        
    except Exception as e:
        logger.error(f"Server error in getAllReciters: {str(e)}")
        return jsonify({
            'error': f'Server error: {str(e)}'
        }), 500

if __name__ == '__main__':
    app.run(debug=True)
