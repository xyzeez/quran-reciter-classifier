"""
Main Flask server for Quran reciter and ayah identification API endpoints.
"""
import logging
from flask import Flask, request, jsonify
import numpy as np
import json
from pathlib import Path
import argparse

from src.utils.audio_utils import process_audio_file
from src.features import extract_features
from server.utils.model_loader import initialize_models, get_reciter_model, get_ayah_model
from src.utils.ayah_matching import find_matching_ayah, load_quran_data
from src.utils.distance_utils import calculate_distances, analyze_prediction_reliability
from server.config import TOP_N_PREDICTIONS, HOST, PORT
from server.utils.text_utils import to_arabic_number

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Load Quran data at startup
quran_data_result = load_quran_data()
if not quran_data_result:
    logger.error("Failed to load Quran data")
    raw_quran_data = []
else:
    raw_quran_data = quran_data_result.get('raw_data', [])

reciter_success, ayah_success = initialize_models()
if not (reciter_success and ayah_success):
    logger.error("Failed to initialize one or more models")

@app.route('/getAyah', methods=['POST'])
def get_ayah():
    """Identify a Quranic verse from an audio recording."""
    try:
        if 'audio' not in request.files:
            return jsonify({
                'error': 'No audio file provided. Please send the file with key "audio" in form data.'
            }), 400

        audio_file = request.files['audio']
        if not audio_file:
            return jsonify({
                'error': 'No audio file selected'
            }), 400

        max_matches = int(request.form.get('max_matches', TOP_N_PREDICTIONS))
        min_confidence = float(request.form.get('min_confidence', 0.70))
        
        result = process_audio_file(audio_file, for_ayah=True)
        if isinstance(result[1], str):
            return jsonify({'error': result[1]}), 400

        audio_data, sample_rate = result
        
        model = get_ayah_model()
        transcribed_text = model.transcribe(audio_data, sample_rate)
        if not transcribed_text:
            return jsonify({'error': 'Failed to transcribe audio'}), 500
            
        matches = find_matching_ayah(
            transcribed_text,
            min_confidence=min_confidence,
            max_matches=max_matches
        )
        
        matches_list = matches.get('matches', [])
        
        formatted_matches = []
        for match in matches_list:
            surah_num = int(match['surah_number'])
            ayah_num = int(match['ayah_number'])
            
            # Get unicode from Quran data
            surah_data = next((s for s in raw_quran_data if s['id'] == surah_num), None)
            unicode_char = surah_data['unicode'] if surah_data else ''
            
            formatted_match = {
                'surah_number': to_arabic_number(surah_num),
                'surah_number_en': surah_num,
                'surah_name': match['surah_name'],
                'surah_name_en': match['surah_name_en'],
                'ayah_number': to_arabic_number(ayah_num),
                'ayah_number_en': ayah_num,
                'ayah_text': match['ayah_text'],
                'confidence_score': float(match['confidence_score']),
                'unicode': unicode_char
            }
            formatted_matches.append(formatted_match)
        
        response = {
            'matches_found': len(formatted_matches) > 0,
            'total_matches': len(formatted_matches),
            'matches': formatted_matches,
            'best_match': formatted_matches[0] if formatted_matches else None
        }
        
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
    """Identify a Quran reciter from an audio recording."""
    try:
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
        
        result = process_audio_file(audio_file, for_ayah=False)
        if isinstance(result[1], str):
            return jsonify({'error': result[1]}), 400

        audio_data, sample_rate = result
        
        model = get_reciter_model()
        if model is None:
            return jsonify({'error': 'Model not initialized'}), 500
            
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
            if features.ndim == 1:
                features = features.reshape(1, -1)

            prediction = model.predict(features)[0]
            probabilities = model.predict_proba(features)[0]

            distances = calculate_distances(features[0], model.centroids)
            reliability = analyze_prediction_reliability(
                probabilities, distances, model.thresholds, prediction)

            sorted_indices = np.argsort(probabilities)[::-1][:TOP_N_PREDICTIONS]
            predictions = []
            
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
                
                reciter_info = reciters_data.get(reciter_name, {})
                servers = reciter_info.get('servers', [])
                server_url = servers[0] if isinstance(servers, list) and servers else ''
                
                predictions.append({
                    'name': reciter_name,
                    'confidence': confidence * 100,
                    'nationality': reciter_info.get('nationality', ''),
                    'serverUrl': server_url,
                    'flagUrl': reciter_info.get('flagUrl', ''),
                    'imageUrl': reciter_info.get('imageUrl', '')
                })

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
    """Get a list of all available reciters."""
    try:
        try:
            reciters_file = Path(__file__).resolve().parent.parent / 'data' / 'reciters.json'
            if not reciters_file.exists():
                return jsonify({'error': 'Reciters data file not found.'}), 500
                
            with open(reciters_file, 'r', encoding='utf-8') as f:
                reciters_data = json.load(f)
        except Exception as e:
            logger.error(f"Error loading reciters data: {str(e)}")
            return jsonify({'error': 'Error loading reciters data.'}), 500

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
    parser = argparse.ArgumentParser(description="Run the Quran Reciter API server.")
    parser.add_argument('--debug', action='store_true', help='Run the server in debug mode.')
    args = parser.parse_args()

    app.run(host=HOST, port=PORT, debug=args.debug)
