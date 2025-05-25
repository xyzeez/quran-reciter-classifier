import argparse
import requests
import time
import json
import csv
from pathlib import Path
from datetime import datetime
import sys
import os
import soundfile as sf
from tqdm import tqdm # Added tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SUPPORTED_AUDIO_EXTS = ['.mp3', '.wav', '.m4a']

# Descriptions for each test case
TC_DESCRIPTIONS = {
    'tc-001': "Reciter ID - Clean Audio",
    'tc-002': "Reciter ID - Non-Training Reciters (Clean Audio)",
    'tc-003': "Ayah ID - Clean Audio",
    'tc-004': "Reciter ID - Noisy Audio",
    'tc-005': "Reciter ID - Non-Training Reciters (Noisy Audio)",
    'tc-006': "Ayah ID - Noisy Audio",
    'tc-007': "Reciter ID - Very Short Audio (Clean)",
    'tc-008': "Reciter ID - Silence/Non-Speech Audio",
    'tc-009': "Ayah ID - Very Short Audio (Clean)",
    'tc-010': "Ayah ID - Silence/Non-Speech Audio",
}

def generate_run_id():
    return datetime.now().strftime('%Y%m%d_%H%M%S') + '_test'

def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def get_test_cases():
    tc_dir = Path('data/test-cases')
    if not tc_dir.exists():
        return []
    return [d.name for d in tc_dir.iterdir() if d.is_dir() and d.name.startswith('tc-')]

def check_server_available(endpoint_url):
    """Check if the server is running by sending a GET request to /health."""
    try:
        resp = requests.get(endpoint_url + '/health')
        if resp.status_code == 200:
            try:
                data = resp.json()
                if data.get('status') == 'ok':
                    return True
                else:
                    print(f"/health responded with status: {data.get('status')}, services: {data.get('services')}")
                    return False
            except Exception as e:
                print(f"/health did not return valid JSON: {e}")
                return False
        else:
            print(f"Server responded with status code {resp.status_code} at {endpoint_url}/health.")
            return False
    except Exception as e:
        print(f"Could not connect to server at {endpoint_url}. Error: {e}")
        return False

def get_model_info(endpoint_url):
    """Query the /models endpoint and return the model info dict (or None on error)."""
    try:
        resp = requests.get(endpoint_url + '/models')
        if resp.status_code == 200:
            return resp.json()
        else:
            print(f"⚠️ Warning: /models responded with status {resp.status_code}")
            return None
    except Exception as e:
        print(f"⚠️ Warning: Could not retrieve model info from {endpoint_url}/models: {e}")
        return None

def run_tc_001(endpoint_url, report_dir, model_info=None):
    """Run TC-001: Reciter ID with high-quality audio."""
    tc_name = 'tc-001'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"⚠️ Test case folder {tc_path} does not exist. Skipping {tc_name}.")
        return
    
    reciter_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    if not reciter_dirs:
        print(f"ℹ️ No reciter directories found in {tc_path}. Skipping {tc_name}.")
        return
        
    results = []
    overall_files_processed = 0
    errors_during_processing = 0

    for reciter_dir in reciter_dirs:
        reciter_name = reciter_dir.name
        audio_files = [f for f in reciter_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        if not audio_files:
            # This message will be shown outside tqdm loop for clarity
            # tqdm.write(f"ℹ️ No audio files found for reciter {reciter_name} in {reciter_dir}.") 
            continue # Skip this reciter if no files
        
        # This print will be outside the tqdm loop for this reciter dir
        print(f"  Processing reciter: {reciter_name} ({len(audio_files)} files)")
        for audio_file in tqdm(audio_files, desc=f"    Files for {reciter_name.ljust(25)}", unit="file", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
            overall_files_processed += 1
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception:
                pass 

            with open(audio_file, 'rb') as f_audio:
                files = {'audio': (audio_file.name, f_audio, 'audio/mpeg')}
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response', 'main_prediction': None, 'reliable': None, 'top_predictions': None}
                        status_code = status_code if status_code else 500 
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None 
                    resp_json = {'error': str(e), 'main_prediction': None, 'reliable': None, 'top_predictions': None}
                
                main_pred = resp_json.get('main_prediction')
                server_response_time_ms = resp_json.get('response_time_ms')
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                is_correct_pred = (main_pred.get('name') == reciter_name) if main_pred and main_pred.get('name') is not None else False
                
                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'reciter_actual': reciter_name,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'reliable': resp_json.get('reliable'),
                    'predicted_reciter': main_pred.get('name') if main_pred else None,
                    'confidence': main_pred.get('confidence') if main_pred else None,
                    'is_correct': is_correct_pred,
                    'error': resp_json.get('error') if 'error' in resp_json else '',
                    'top_predictions': resp_json.get('top_predictions')
                }
                results.append(result)
                
                if result['error']:
                    tqdm.write(f"    🛑 ERROR processing {audio_file.name}: {result['error']} (Status: {status_code})")
                    errors_during_processing +=1
                elif status_code != 200:
                    tqdm.write(f"    🛑 SERVER ERROR for {audio_file.name}: Status {status_code}, Response: {resp_json.get('error', 'No error detail')}")
                    errors_during_processing +=1
                elif not result['is_correct']:
                    predicted_reciter_str = result['predicted_reciter'] if result['predicted_reciter'] else "None"
                    confidence_str = f"{result['confidence']:.2f}%" if result['confidence'] is not None else "N/A"
                    tqdm.write(f"    ⚠️ MISCLASSIFIED: {audio_file.name} (Actual: {reciter_name}) -> Pred: {predicted_reciter_str}, Conf: {confidence_str}, Reliable: {result['reliable']}")

    if not results and overall_files_processed == 0 and errors_during_processing == 0:
        print(f"  ℹ️ No audio files processed for {tc_name}. Please check data folder: {tc_path}")
        return 

    ensure_dir(report_dir)
    # For TC-001, the original filenames were generic. Keeping them for this specific TC for now.
    csv_path = report_dir / 'results.csv' 
    json_path = report_dir / 'results.json' 
    summary_txt_path = report_dir / f'summary_{tc_name}.txt' # Use tc_name for summary consistency
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'reciter_actual', 'predicted_reciter', 'confidence', 
                      'reliable', 'is_correct', 'client_round_trip_time_s', 'server_response_time_s',
                      'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {
        # model_info is no longer included here, it's at the run level
        'results': results
    }
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-001] Report saved: {csv_path} and {json_path}")

    # --- Generate TXT Summary --- 
    total_files = len(results)
    correct_predictions = sum(1 for r in results if r['is_correct'])
    incorrect_predictions = total_files - correct_predictions
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

    reliable_count = sum(1 for r in results if r['reliable'])
    unreliable_count = total_files - reliable_count

    conf_correct = [r['confidence'] for r in results if r['is_correct'] and r['confidence'] is not None]
    avg_conf_correct = (sum(conf_correct) / len(conf_correct)) if conf_correct else 0

    conf_incorrect = [r['confidence'] for r in results if not r['is_correct'] and r['confidence'] is not None]
    avg_conf_incorrect = (sum(conf_incorrect) / len(conf_incorrect)) if conf_incorrect else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files = [r['file'] for r in results if r['error']]

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-001 Summary\n")
        sf_txt.write(f"--------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correct Predictions: {correct_predictions} ({accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrect Predictions: {incorrect_predictions}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Reliable Predictions: {reliable_count}\n")
        sf_txt.write(f"Unreliable Predictions: {unreliable_count}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct): {avg_conf_correct:.2f}%\n" if conf_correct else "Avg. Confidence (Correct): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect): {avg_conf_incorrect:.2f}%\n" if conf_incorrect else "Avg. Confidence (Incorrect): N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files)}):\n")
            for err_file in error_files:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing.\n")
    
    print(f"[TC-001] TXT Summary saved: {summary_txt_path}")

def run_tc_002(endpoint_url, report_dir, model_info=None):
    """Run TC-002: Non-training reciter handling."""
    tc_name = 'tc-002'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"⚠️ Test case folder {tc_path} does not exist. Skipping {tc_name}.")
        return
    
    unknown_reciter_sample_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    if not unknown_reciter_sample_dirs:
        print(f"  ℹ️ No sample directories found in {tc_path}. Skipping {tc_name}.")
        return
        
    results = []
    overall_files_processed = 0
    errors_during_processing = 0

    for sample_dir in unknown_reciter_sample_dirs:
        source_description = sample_dir.name 
        audio_files = [f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        if not audio_files:
            # tqdm.write(f"  ℹ️ No audio files found in sample directory {sample_dir}.") # This would need tqdm instance if used before loop
            continue
        
        print(f"  Processing sample category: {source_description} ({len(audio_files)} files)")
        for audio_file in tqdm(audio_files, desc=f"    Files for {source_description.ljust(25)}", unit="file", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
            overall_files_processed += 1
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception:
                pass

            with open(audio_file, 'rb') as f_audio:
                files = {'audio': (audio_file.name, f_audio, 'audio/mpeg')}
                payload = {'show_unreliable_predictions': 'true'} 
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files, data=payload)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response', 'main_prediction': None, 'reliable': None}
                        status_code = status_code if status_code else 500
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e), 'main_prediction': None, 'reliable': None}
                
                main_pred = resp_json.get('main_prediction')
                server_response_time_ms = resp_json.get('response_time_ms')
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                is_reliable_response = resp_json.get('reliable')
                is_correct_handling = is_reliable_response is False # Correct if deemed unreliable

                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'source_description': source_description, 
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'server_reliable_flag': is_reliable_response,
                    'predicted_reciter_if_any': main_pred.get('name') if main_pred else None,
                    'confidence_if_any': main_pred.get('confidence') if main_pred else None,
                    'correctly_handled_as_unreliable': is_correct_handling,
                    'error': resp_json.get('error') if 'error' in resp_json else '',
                }
                results.append(result)

                if result['error']:
                    tqdm.write(f"    🛑 ERROR processing {audio_file.name}: {result['error']} (Status: {status_code})")
                    errors_during_processing +=1
                elif status_code != 200:
                    tqdm.write(f"    🛑 SERVER ERROR for {audio_file.name}: Status {status_code}, Response: {resp_json.get('error', 'No error detail')}")
                    errors_during_processing +=1
                elif not result['correctly_handled_as_unreliable']:
                    # This is a False Positive for TC-002 (marked reliable when it should be unreliable)
                    pred_reciter = result['predicted_reciter_if_any'] or "None"
                    pred_conf = f"{result['confidence_if_any']:.2f}%" if result['confidence_if_any'] is not None else "N/A"
                    tqdm.write(f"    ⚠️ FALSE POSITIVE (Reliable): {audio_file.name} (Source: {source_description}) -> Pred: {pred_reciter}, Conf: {pred_conf}")
                # Optional: Log if reliable flag is missing but no error
                # elif is_reliable_response is None and not result['error']:
                #     tqdm.write(f"    ⚠️ UNKNOWN HANDLING: {audio_file.name} (Reliable flag missing, Source: {source_description})")

    if not results and overall_files_processed == 0 and errors_during_processing == 0:
        print(f"  ℹ️ No audio files processed for {tc_name}. Please check data folder: {tc_path}")
        return

    ensure_dir(report_dir)
    csv_path = report_dir / f'results_{tc_name}.csv'
    json_path = report_dir / f'results_{tc_name}.json'
    summary_txt_path = report_dir / f'summary_{tc_name}.txt'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'source_description', 'server_reliable_flag', 
                      'predicted_reciter_if_any', 'confidence_if_any', 'correctly_handled_as_unreliable', 
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-002] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-002 --- 
    total_files = len(results)
    correctly_handled_count = sum(1 for r in results if r['correctly_handled_as_unreliable'])
    false_positive_count = total_files - correctly_handled_count # Files that were reliably (mis)identified
    
    handling_accuracy = (correctly_handled_count / total_files) * 100 if total_files > 0 else 0
    false_positive_rate = (false_positive_count / total_files) * 100 if total_files > 0 else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files_tc002 = [r['file'] for r in results if r['error']]

    # Confidence for false positives (i.e., server_reliable_flag was True)
    conf_false_positives = [r['confidence_if_any'] for r in results if r['server_reliable_flag'] is True and r['confidence_if_any'] is not None]
    avg_conf_false_positive = (sum(conf_false_positives) / len(conf_false_positives)) if conf_false_positives else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-002 Summary: Non-training reciter handling\n")
        sf_txt.write(f"--------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correctly Handled (Marked Unreliable by Server): {correctly_handled_count} ({handling_accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrectly Handled (False Positives - Marked Reliable): {false_positive_count} ({false_positive_rate:.2f}%)\n")
        sf_txt.write(f"Avg. Confidence of False Positive IDs: {avg_conf_false_positive:.2f}%\n" if conf_false_positives else "Avg. Confidence of False Positive IDs: N/A (No false positives or no confidence reported)\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files_tc002:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files_tc002)}):\n")
            for err_file in error_files_tc002:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing.\n")
    
    print(f"[TC-002] TXT Summary saved: {summary_txt_path}")

def run_tc_003(endpoint_url, report_dir, model_info=None):
    """Run TC-003: Ayah Identification."""
    tc_name = 'tc-003'
    # Description is fetched in main
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"⚠️ Test case folder {tc_path} does not exist. Skipping {tc_name}.")
        return

    audio_files_all = [f for f in tc_path.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
    audio_files_valid_format = []
    # Pre-filter for SSSAAA format and alert about invalid ones
    for f in audio_files_all:
        if len(f.stem) == 6 and f.stem.isdigit():
            audio_files_valid_format.append(f)
        else:
            print(f"  ⚠️ SKIPPING (Invalid Filename): {f.name} (Expected SSSAAA format for Ayah ID tests).")

    if not audio_files_valid_format:
        print(f"  ℹ️ No audio files with SSSAAA format found in {tc_path}. Skipping {tc_name}.")
        return

    results = []
    overall_files_processed = 0
    errors_during_processing = 0

    print(f"  Processing Ayah ID files ({len(audio_files_valid_format)} files)")
    for audio_file in tqdm(audio_files_valid_format, desc=f"    Files for {tc_name.ljust(25)}", unit="file", leave=False, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]'):
        overall_files_processed += 1
        filename_stem = audio_file.stem
        # Already validated SSSAAA format, so direct parse should be safe
        expected_surah = int(filename_stem[:3])
        expected_ayah = int(filename_stem[3:])

        duration = None
        try:
            duration = sf.info(str(audio_file)).duration
        except Exception:
            pass

        with open(audio_file, 'rb') as f_audio:
            files = {'audio': (audio_file.name, f_audio, 'audio/mpeg')}
            payload = {} 
            start_time = time.time()
            try:
                response = requests.post(endpoint_url + '/getAyah', files=files, data=payload)
                elapsed = time.time() - start_time
                status_code = response.status_code
                try:
                    resp_json = response.json()
                except Exception:
                    resp_json = {'error': 'Invalid JSON response', 'best_match': None, 'matches': [], 'transcription': ''}
                    status_code = status_code if status_code else 500
            except Exception as e:
                elapsed = time.time() - start_time
                status_code = None
                resp_json = {'error': str(e), 'best_match': None, 'matches': [], 'transcription': ''}
            
            server_response_time_ms = resp_json.get('response_time_ms')
            server_duration_s = None
            if server_response_time_ms is not None:
                server_duration_s = server_response_time_ms / 1000.0

            best_match_data = resp_json.get('best_match')
            matches_list = resp_json.get('matches', [])
            transcription = resp_json.get('transcription', '') or resp_json.get('asr_transcription', '')
            
            top_match_correct = False
            found_in_matches = False
            top_match_details = {}

            if best_match_data: 
                top_match_details = {
                    'surah': best_match_data.get('surah_number_en'),
                    'ayah': best_match_data.get('ayah_number_en'),
                    'text': best_match_data.get('ayah_text'),
                    'confidence': best_match_data.get('confidence_score')
                }
                if top_match_details['surah'] == expected_surah and top_match_details['ayah'] == expected_ayah:
                    top_match_correct = True
            
            for match_item in matches_list:
                if match_item.get('surah_number_en') == expected_surah and match_item.get('ayah_number_en') == expected_ayah:
                    found_in_matches = True
                    break
            
            result = {
                'file': str(audio_file),
                'duration_seconds': duration,
                'expected_surah': expected_surah,
                'expected_ayah': expected_ayah,
                'status_code': status_code,
                'client_round_trip_time_s': elapsed,
                'server_response_time_s': server_duration_s,
                'transcription': transcription,
                'top_match_surah': top_match_details.get('surah'),
                'top_match_ayah': top_match_details.get('ayah'),
                'top_match_text': top_match_details.get('text'),
                'top_match_confidence': top_match_details.get('confidence'),
                'top_match_correct': top_match_correct,
                'found_in_matches': found_in_matches,
                'num_matches_returned': len(matches_list),
                'all_matches': matches_list,
                'error': resp_json.get('error') if 'error' in resp_json else '',
            }
            results.append(result)

            if result['error']:
                tqdm.write(f"    🛑 ERROR processing {audio_file.name}: {result['error']} (Status: {status_code})")
                errors_during_processing +=1
            elif status_code != 200:
                tqdm.write(f"    🛑 SERVER ERROR for {audio_file.name}: Status {status_code}, Response: {resp_json.get('error', 'No error detail')}")
                errors_during_processing +=1
            elif not top_match_correct and found_in_matches:
                 # Log if top match is wrong, but expected is in the list
                top_s = top_match_details.get('surah','-'); top_a = top_match_details.get('ayah','-')
                top_conf = f"{top_match_details.get('confidence', 0):.2f}%"
                tqdm.write(f"    ℹ️  FOUND IN LIST (Not Top): {audio_file.name} (Exp: {expected_surah}:{expected_ayah}) -> Top was {top_s}:{top_a} ({top_conf})")
            elif not top_match_correct and not found_in_matches and matches_list:
                # Log if incorrect and not in list, but some matches were returned
                top_s = top_match_details.get('surah','-'); top_a = top_match_details.get('ayah','-')
                top_conf = f"{top_match_details.get('confidence', 0):.2f}%"
                tqdm.write(f"    ⚠️ INCORRECT MATCH: {audio_file.name} (Exp: {expected_surah}:{expected_ayah}) -> Top was {top_s}:{top_a} ({top_conf})")
            elif not matches_list and not result['error']:
                 # Log if no matches were returned at all
                tqdm.write(f"    ⚠️ NO MATCHES RETURNED for {audio_file.name} (Exp: {expected_surah}:{expected_ayah}). Transcription: '{transcription[:30]}...'")
    
    if not results and overall_files_processed == 0 and errors_during_processing == 0:
        print(f"  ℹ️ No valid SSSAAA audio files processed for {tc_name}. Please check data folder: {tc_path}")
        return

    # ensure_dir(report_dir) # Already done in main loop
    csv_path = report_dir / f'results_{tc_name}.csv'
    json_path = report_dir / f'results_{tc_name}.json'
    summary_txt_path = report_dir / f'summary_{tc_name}.txt'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'expected_surah', 'expected_ayah', 
                      'top_match_surah', 'top_match_ayah', 'top_match_text', 'top_match_confidence', 
                      'top_match_correct', 'found_in_matches', 'num_matches_returned', 'transcription',
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-003] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-003 --- 
    total_files = len(results)
    errors_count_tc003 = sum(1 for r in results if r['error'])
    files_processed_successfully = total_files - errors_count_tc003
    
    top_match_correct_count = sum(1 for r in results if r['top_match_correct'])
    found_in_matches_count = sum(1 for r in results if r['found_in_matches'] and not r['top_match_correct']) # only count if not already top_match
    
    accuracy_top_match = (top_match_correct_count / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0
    accuracy_found_in_list = ((top_match_correct_count + found_in_matches_count) / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0

    conf_correct_top = [r['top_match_confidence'] for r in results if r['top_match_correct'] and r['top_match_confidence'] is not None]
    avg_conf_correct_top = (sum(conf_correct_top) / len(conf_correct_top)) if conf_correct_top else 0

    conf_incorrect_top_but_match_returned = [
        r['top_match_confidence'] for r in results 
        if not r['top_match_correct'] and r['num_matches_returned'] > 0 and r['top_match_confidence'] is not None
    ]
    avg_conf_incorrect_top = (sum(conf_incorrect_top_but_match_returned) / len(conf_incorrect_top_but_match_returned)) if conf_incorrect_top_but_match_returned else 0

    num_matches_returned_list = [r['num_matches_returned'] for r in results if not r['error']]
    avg_matches_returned = (sum(num_matches_returned_list) / len(num_matches_returned_list)) if num_matches_returned_list else 0 # Corrected: was using num_matches_returned_list for check but not sum/len

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None and not r['error']]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-003 Summary: Ayah Identification\n")
        sf_txt.write(f"----------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Submitted: {total_files}\n")
        sf_txt.write(f"Files Processed Successfully (no errors): {files_processed_successfully}\n")
        sf_txt.write(f"Files with Errors: {errors_count_tc003}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Accuracy (Top Match Correct): {top_match_correct_count}/{files_processed_successfully} ({accuracy_top_match:.2f}%)\n")
        sf_txt.write(f"Accuracy (Found in Returned List, incl. top): {(top_match_correct_count + found_in_matches_count)}/{files_processed_successfully} ({accuracy_found_in_list:.2f}%)\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct Top Match): {avg_conf_correct_top:.2f}\n" if conf_correct_top else "Avg. Confidence (Correct Top Match): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect Top Match, but matches returned): {avg_conf_incorrect_top:.2f}\n" if conf_incorrect_top_but_match_returned else "Avg. Confidence (Incorrect Top Match): N/A\n")
        sf_txt.write(f"Avg. Number of Matches Returned per File: {avg_matches_returned:.2f}\n" if num_matches_returned_list else "Avg. Number of Matches Returned per File: N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time (Successful Files): {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if errors_count_tc003 > 0:
            sf_txt.write(f"Files with Errors Details ({errors_count_tc003}):\n")
            for r_err in results:
                if r_err['error']:
                    sf_txt.write(f"  - {r_err['file']}: {r_err['error']}\n")
        else:
            sf_txt.write("No errors reported during Ayah identification processing.\n")

    print(f"[TC-003] TXT Summary saved: {summary_txt_path}")

def run_tc_004(endpoint_url, report_dir, model_info=None):
    """Run TC-004: Reciter ID with noisy audio."""
    tc_name = 'tc-004'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-004.")
        return
    
    print(f"\nRunning Test Case: {tc_name} - Reciter ID with noisy audio")

    reciter_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    results = []
    for reciter_dir in reciter_dirs:
        reciter_name = reciter_dir.name
        audio_files = [f for f in reciter_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        for audio_file in audio_files:
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file.name}: {e}")

            with open(audio_file, 'rb') as f:
                files = {'audio': (audio_file.name, f, 'audio/mpeg')}
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response'}
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e)}
                
                main_pred = resp_json.get('main_prediction') if isinstance(resp_json, dict) else None
                server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'reciter_actual': reciter_name,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'reliable': resp_json.get('reliable') if isinstance(resp_json, dict) else None,
                    'predicted_reciter': main_pred.get('name') if main_pred else None,
                    'confidence': main_pred.get('confidence') if main_pred else None,
                    'is_correct': (main_pred.get('name') == reciter_name) if main_pred else False,
                    'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
                    'top_predictions': resp_json.get('top_predictions') if isinstance(resp_json, dict) else None
                }
                results.append(result)
                duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
                confidence_str = f"{result['confidence']:.2f}%" if result['confidence'] is not None else "N/A"
                server_duration_str = f"{server_duration_s:.4f}s" if server_duration_s is not None else "N/A"
                
                print(f"[TC-004] {audio_file.name}: Duration={duration_str}, Pred={result['predicted_reciter']} " \
                      f"(Actual={reciter_name}), Conf={confidence_str}, Reliable={result['reliable']}, ServerResponseTime={server_duration_str}, " \
                      f"ClientTime={elapsed:.2f}s, Status={status_code}")
    
    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-004.csv' # Changed filename
    json_path = report_dir / 'results_tc-004.json' # Changed filename
    summary_txt_path = report_dir / 'summary_tc-004.txt' # Changed filename
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'reciter_actual', 'predicted_reciter', 'confidence', 
                      'reliable', 'is_correct', 'client_round_trip_time_s', 'server_response_time_s',
                      'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-004] Reports saved: {csv_path} and {json_path}")

    # --- Generate TXT Summary for TC-004 (similar to TC-001) --- 
    total_files = len(results)
    correct_predictions = sum(1 for r in results if r['is_correct'])
    incorrect_predictions = total_files - correct_predictions
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

    reliable_count = sum(1 for r in results if r['reliable'])
    unreliable_count = total_files - reliable_count

    conf_correct = [r['confidence'] for r in results if r['is_correct'] and r['confidence'] is not None]
    avg_conf_correct = (sum(conf_correct) / len(conf_correct)) if conf_correct else 0

    conf_incorrect = [r['confidence'] for r in results if not r['is_correct'] and r['confidence'] is not None]
    avg_conf_incorrect = (sum(conf_incorrect) / len(conf_incorrect)) if conf_incorrect else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files = [r['file'] for r in results if r['error']]

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-004 Summary: Reciter ID with Noisy Audio\n") # Changed title
        sf_txt.write(f"-------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correct Predictions: {correct_predictions} ({accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrect Predictions: {incorrect_predictions}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Reliable Predictions: {reliable_count}\n")
        sf_txt.write(f"Unreliable Predictions: {unreliable_count}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct): {avg_conf_correct:.2f}%\n" if conf_correct else "Avg. Confidence (Correct): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect): {avg_conf_incorrect:.2f}%\n" if conf_incorrect else "Avg. Confidence (Incorrect): N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files)}):\n")
            for err_file in error_files:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing.\n")
    
    print(f"[TC-004] TXT Summary saved: {summary_txt_path}")

def run_tc_005(endpoint_url, report_dir, model_info=None):
    """Run TC-005: Non-training reciter handling with noisy audio."""
    tc_name = 'tc-005'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-005.")
        return
    
    unknown_reciter_noisy_sample_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    results = []

    print(f"\nRunning Test Case: {tc_name} - Non-training reciter handling with noisy audio")

    for sample_dir in unknown_reciter_noisy_sample_dirs:
        source_description = sample_dir.name 
        audio_files = [f for f in sample_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        
        for audio_file in audio_files:
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file.name}: {e}")

            with open(audio_file, 'rb') as f:
                files = {'audio': (audio_file.name, f, 'audio/mpeg')}
                payload = {'show_unreliable_predictions': 'true'} 
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files, data=payload)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response'}
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e)}
                
                main_pred = resp_json.get('main_prediction') if isinstance(resp_json, dict) else None
                server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                is_reliable_response = resp_json.get('reliable') if isinstance(resp_json, dict) else None
                is_correct_handling = is_reliable_response is False # Correct if deemed unreliable

                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'source_description': source_description,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'server_reliable_flag': is_reliable_response,
                    'predicted_reciter_if_any': main_pred.get('name') if main_pred else None,
                    'confidence_if_any': main_pred.get('confidence') if main_pred else None,
                    'correctly_handled_as_unreliable': is_correct_handling,
                    'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
                }
                results.append(result)
                duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
                confidence_str = f"{result['confidence_if_any']:.2f}%" if result['confidence_if_any'] is not None else "N/A"
                server_duration_str = f"{server_duration_s:.4f}s" if server_duration_s is not None else "N/A"
                handling_status = "Correctly Unreliable" if is_correct_handling else "False Positive (Reliable)"
                if is_reliable_response is None and not result['error']:
                    handling_status = "Unknown (Reliable flag missing)"
                elif result['error']:
                    handling_status = f"Error: {result['error']}"

                print(f"[TC-005] {audio_file.name} (Source: {source_description}): Handling={handling_status}, " \
                      f"Pred={result['predicted_reciter_if_any']}, Conf={confidence_str}, " \
                      f"ServerResponseTime={server_duration_str}, ClientTime={elapsed:.2f}s, Status={status_code}")
    
    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-005.csv' # Changed filename
    json_path = report_dir / 'results_tc-005.json' # Changed filename
    summary_txt_path = report_dir / 'summary_tc-005.txt' # Changed filename
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'source_description', 'server_reliable_flag', 
                      'predicted_reciter_if_any', 'confidence_if_any', 'correctly_handled_as_unreliable', 
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-005] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-005 (similar to TC-002) --- 
    total_files = len(results)
    correctly_handled_count = sum(1 for r in results if r['correctly_handled_as_unreliable'])
    false_positive_count = total_files - correctly_handled_count
    
    handling_accuracy = (correctly_handled_count / total_files) * 100 if total_files > 0 else 0
    false_positive_rate = (false_positive_count / total_files) * 100 if total_files > 0 else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files_tc005 = [r['file'] for r in results if r['error']]

    conf_false_positives = [r['confidence_if_any'] for r in results if r['server_reliable_flag'] is True and r['confidence_if_any'] is not None]
    avg_conf_false_positive = (sum(conf_false_positives) / len(conf_false_positives)) if conf_false_positives else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-005 Summary: Non-training reciter handling with Noisy Audio\n") # Changed title
        sf_txt.write(f"-------------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correctly Handled (Marked Unreliable): {correctly_handled_count} ({handling_accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrectly Handled (Marked Reliable - False Positive ID): {false_positive_count} ({false_positive_rate:.2f}%)\n")
        sf_txt.write(f"Avg. Confidence of False Positive IDs: {avg_conf_false_positive:.2f}%\n" if conf_false_positives else "Avg. Confidence of False Positive IDs: N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files_tc005:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files_tc005)}):\n")
            for err_file in error_files_tc005:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing of silence/non-speech audio.\n")
    
    print(f"[TC-005] TXT Summary saved: {summary_txt_path}")

def run_tc_006(endpoint_url, report_dir, model_info=None):
    """Run TC-006: Ayah Identification with noisy audio."""
    tc_name = 'tc-006'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-006.")
        return

    # Expect filenames like SSSAAA.ext or SSSAAA_description.ext for noisy Ayah audio
    # We will parse the SSSAAA part.
    audio_files = []
    for f in tc_path.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            if len(f.stem) >= 6 and f.stem[:6].isdigit(): # Basic check for SSSAAA prefix
                audio_files.append(f)
            else:
                print(f"[TC-006] Warning: File {f.name} does not match SSSAAA naming pattern prefix. Skipping.")
    
    results = []
    print(f"\nRunning Test Case: {tc_name} - Ayah Identification with noisy audio")

    for audio_file in audio_files:
        filename_stem_prefix = audio_file.stem[:6] # Take the first 6 chars for SSSAAA
        try:
            expected_surah = int(filename_stem_prefix[:3])
            expected_ayah = int(filename_stem_prefix[3:])
        except ValueError:
            print(f"Warning: Could not parse Surah/Ayah from filename {audio_file.name} (using prefix {filename_stem_prefix}). Skipping.")
            continue

        duration = None
        try:
            duration = sf.info(str(audio_file)).duration
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_file.name}: {e}")

        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file.name, f, 'audio/mpeg')}
            payload = {} # Use server defaults for Ayah identification parameters
            start_time = time.time()
            try:
                response = requests.post(endpoint_url + '/getAyah', files=files, data=payload)
                elapsed = time.time() - start_time
                status_code = response.status_code
                try:
                    resp_json = response.json()
                except Exception:
                    resp_json = {'error': 'Invalid JSON response'}
            except Exception as e:
                elapsed = time.time() - start_time
                status_code = None
                resp_json = {'error': str(e)}
            
            server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
            server_duration_s = None
            if server_response_time_ms is not None:
                server_duration_s = server_response_time_ms / 1000.0

            best_match_data = resp_json.get('best_match') if isinstance(resp_json, dict) else None
            matches_list = resp_json.get('matches', []) if isinstance(resp_json, dict) else []
            transcription = resp_json.get('transcription', '') if isinstance(resp_json, dict) else resp_json.get('asr_transcription', '')
            
            top_match_correct = False
            found_in_matches = False
            top_match_details = {}

            if best_match_data:
                top_match_details = {
                    'surah': best_match_data.get('surah_number_en'),
                    'ayah': best_match_data.get('ayah_number_en'),
                    'text': best_match_data.get('ayah_text'),
                    'confidence': best_match_data.get('confidence_score')
                }
                if top_match_details['surah'] == expected_surah and top_match_details['ayah'] == expected_ayah:
                    top_match_correct = True
            
            for match_item in matches_list:
                if match_item.get('surah_number_en') == expected_surah and match_item.get('ayah_number_en') == expected_ayah:
                    found_in_matches = True
                    break
            
            result = {
                'file': str(audio_file),
                'duration_seconds': duration,
                'expected_surah': expected_surah,
                'expected_ayah': expected_ayah,
                'status_code': status_code,
                'client_round_trip_time_s': elapsed,
                'server_response_time_s': server_duration_s,
                'transcription': transcription,
                'top_match_surah': top_match_details.get('surah'),
                'top_match_ayah': top_match_details.get('ayah'),
                'top_match_text': top_match_details.get('text'),
                'top_match_confidence': top_match_details.get('confidence'),
                'top_match_correct': top_match_correct,
                'found_in_matches': found_in_matches,
                'num_matches_returned': len(matches_list),
                'all_matches': matches_list,
                'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
            }
            results.append(result)

            status_msg = "Correct (Top)" if top_match_correct else (
                "Correct (In List)" if found_in_matches else "Incorrect"
            )
            if result['error']:
                status_msg = f"Error: {result['error']}"
            elif not matches_list and not result['error']:
                status_msg = "No Matches Returned"

            conf_str = f"{top_match_details.get('confidence'):.2f}" if top_match_details.get('confidence') is not None else "N/A"
            print(f"[TC-006] {audio_file.name} (Exp: {expected_surah}:{expected_ayah}): Status={status_msg}, " \
                  f"TopMatch=({top_match_details.get('surah', '-')}:{top_match_details.get('ayah', '-')}, Conf={conf_str}), " \
                  f"Matches Returned={len(matches_list)}, ServerTime={server_duration_s if server_duration_s is not None else 'N/A'}")

    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-006.csv' # Changed filename
    json_path = report_dir / 'results_tc-006.json' # Changed filename
    summary_txt_path = report_dir / 'summary_tc-006.txt' # Changed filename

    csv_fieldnames = ['file', 'duration_seconds', 'expected_surah', 'expected_ayah', 
                      'top_match_surah', 'top_match_ayah', 'top_match_text', 'top_match_confidence', 
                      'top_match_correct', 'found_in_matches', 'num_matches_returned', 'transcription',
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in csv_fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-006] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-006 (similar to TC-003) --- 
    total_files = len(results)
    errors_count_tc006 = sum(1 for r in results if r['error'])
    files_processed_successfully = total_files - errors_count_tc006
    
    top_match_correct_count = sum(1 for r in results if r['top_match_correct'])
    found_in_matches_count = sum(1 for r in results if r['found_in_matches'] and not r['top_match_correct'])
    
    accuracy_top_match = (top_match_correct_count / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0
    accuracy_found_in_list = ((top_match_correct_count + found_in_matches_count) / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0

    conf_correct_top = [r['top_match_confidence'] for r in results if r['top_match_correct'] and r['top_match_confidence'] is not None]
    avg_conf_correct_top = (sum(conf_correct_top) / len(conf_correct_top)) if conf_correct_top else 0

    conf_incorrect_top_but_match_returned = [
        r['top_match_confidence'] for r in results 
        if not r['top_match_correct'] and r['num_matches_returned'] > 0 and r['top_match_confidence'] is not None
    ]
    avg_conf_incorrect_top = (sum(conf_incorrect_top_but_match_returned) / len(conf_incorrect_top_but_match_returned)) if conf_incorrect_top_but_match_returned else 0

    num_matches_returned_list = [r['num_matches_returned'] for r in results if not r['error']]
    avg_matches_returned = (sum(num_matches_returned_list) / len(num_matches_returned_list)) if num_matches_returned_list else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None and not r['error']]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-006 Summary: Ayah Identification with Noisy Audio\n") # Changed title
        sf_txt.write(f"------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Submitted: {total_files}\n")
        sf_txt.write(f"Files Processed Successfully (no errors): {files_processed_successfully}\n")
        sf_txt.write(f"Files with Errors: {errors_count_tc006}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Accuracy (Top Match Correct): {top_match_correct_count}/{files_processed_successfully} ({accuracy_top_match:.2f}%)\n")
        sf_txt.write(f"Accuracy (Found in Returned List, incl. top): {(top_match_correct_count + found_in_matches_count)}/{files_processed_successfully} ({accuracy_found_in_list:.2f}%)\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct Top Match): {avg_conf_correct_top:.2f}\n" if conf_correct_top else "Avg. Confidence (Correct Top Match): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect Top Match, but matches returned): {avg_conf_incorrect_top:.2f}\n" if conf_incorrect_top_but_match_returned else "Avg. Confidence (Incorrect Top Match): N/A\n")
        sf_txt.write(f"Avg. Number of Matches Returned per File: {avg_matches_returned:.2f}\n" if num_matches_returned_list else "Avg. Number of Matches Returned per File: N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time (Successful Files): {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if errors_count_tc006 > 0:
            sf_txt.write(f"Files with Errors Details ({errors_count_tc006}):\n")
            for r_err in results:
                if r_err['error']:
                    sf_txt.write(f"  - {r_err['file']}: {r_err['error']}\n")
        else:
            sf_txt.write("No errors reported during Ayah identification processing with noisy audio.\n") # Changed message

    print(f"[TC-006] TXT Summary saved: {summary_txt_path}")

def run_tc_007(endpoint_url, report_dir, model_info=None):
    """Run TC-007: Reciter ID with very short audio clips."""
    tc_name = 'tc-007'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-007.")
        return
    
    print(f"\nRunning Test Case: {tc_name} - Reciter ID with very short audio clips")

    reciter_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    results = []
    for reciter_dir in reciter_dirs:
        reciter_name = reciter_dir.name
        audio_files = [f for f in reciter_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        for audio_file in audio_files:
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
                if duration > 3.5: # Example threshold, can be adjusted
                    print(f"[TC-007] Warning: File {audio_file.name} is {duration:.2f}s, may not be a 'very short' clip. Proceeding anyway.")
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file.name}: {e}")

            with open(audio_file, 'rb') as f:
                files = {'audio': (audio_file.name, f, 'audio/mpeg')}
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response'}
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e)}
                
                main_pred = resp_json.get('main_prediction') if isinstance(resp_json, dict) else None
                server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'reciter_actual': reciter_name,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'reliable': resp_json.get('reliable') if isinstance(resp_json, dict) else None,
                    'predicted_reciter': main_pred.get('name') if main_pred else None,
                    'confidence': main_pred.get('confidence') if main_pred else None,
                    'is_correct': (main_pred.get('name') == reciter_name) if main_pred else False,
                    'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
                    'top_predictions': resp_json.get('top_predictions') if isinstance(resp_json, dict) else None
                }
                results.append(result)
                duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
                confidence_str = f"{result['confidence']:.2f}%" if result['confidence'] is not None else "N/A"
                server_duration_str = f"{server_duration_s:.4f}s" if server_duration_s is not None else "N/A"
                
                print(f"[TC-007] {audio_file.name}: Duration={duration_str}, Pred={result['predicted_reciter']} " \
                      f"(Actual={reciter_name}), Conf={confidence_str}, Reliable={result['reliable']}, ServerResponseTime={server_duration_str}, " \
                      f"ClientTime={elapsed:.2f}s, Status={status_code}")
    
    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-007.csv'
    json_path = report_dir / 'results_tc-007.json'
    summary_txt_path = report_dir / 'summary_tc-007.txt'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'reciter_actual', 'predicted_reciter', 'confidence', 
                      'reliable', 'is_correct', 'client_round_trip_time_s', 'server_response_time_s',
                      'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-007] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-007 (similar to TC-001) --- 
    total_files = len(results)
    correct_predictions = sum(1 for r in results if r['is_correct'])
    incorrect_predictions = total_files - correct_predictions
    accuracy = (correct_predictions / total_files) * 100 if total_files > 0 else 0

    reliable_count = sum(1 for r in results if r['reliable'])
    unreliable_count = total_files - reliable_count

    conf_correct = [r['confidence'] for r in results if r['is_correct'] and r['confidence'] is not None]
    avg_conf_correct = (sum(conf_correct) / len(conf_correct)) if conf_correct else 0

    conf_incorrect = [r['confidence'] for r in results if not r['is_correct'] and r['confidence'] is not None]
    avg_conf_incorrect = (sum(conf_incorrect) / len(conf_incorrect)) if conf_incorrect else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files = [r['file'] for r in results if r['error']]

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-007 Summary: Reciter ID with Very Short Audio Clips\n")
        sf_txt.write(f"------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correct Predictions: {correct_predictions} ({accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrect Predictions: {incorrect_predictions}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Reliable Predictions: {reliable_count}\n")
        sf_txt.write(f"Unreliable Predictions: {unreliable_count}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct): {avg_conf_correct:.2f}%\n" if conf_correct else "Avg. Confidence (Correct): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect): {avg_conf_incorrect:.2f}%\n" if conf_incorrect else "Avg. Confidence (Incorrect): N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files)}):\n")
            for err_file in error_files:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing of very short clips.\n")
    
    print(f"[TC-007] TXT Summary saved: {summary_txt_path}")

def run_tc_008(endpoint_url, report_dir, model_info=None):
    """Run TC-008: Reciter ID with silence or non-speech audio."""
    tc_name = 'tc-008'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-008.")
        return
    
    sound_type_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    results = []

    print(f"\nRunning Test Case: {tc_name} - Reciter ID with silence or non-speech audio")

    for sound_dir in sound_type_dirs:
        source_description = sound_dir.name # e.g., "Silence", "Door_Slam"
        audio_files = [f for f in sound_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        
        for audio_file in audio_files:
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file.name}: {e}")

            with open(audio_file, 'rb') as f:
                files = {'audio': (audio_file.name, f, 'audio/mpeg')}
                # We want to see predictions, even if unreliable, to spot false positives.
                payload = {'show_unreliable_predictions': 'true'} 
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getReciter', files=files, data=payload)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response'}
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e)}
                
                main_pred = resp_json.get('main_prediction') if isinstance(resp_json, dict) else None
                server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0
                
                is_reliable_response = resp_json.get('reliable') if isinstance(resp_json, dict) else None
                # For TC-008, a "correct" outcome is when the server deems the prediction UNRELIABLE for non-speech/silence.
                is_correct_handling = is_reliable_response is False

                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'source_description': source_description,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'server_reliable_flag': is_reliable_response,
                    'predicted_reciter_if_any': main_pred.get('name') if main_pred else None,
                    'confidence_if_any': main_pred.get('confidence') if main_pred else None,
                    'correctly_handled_as_unreliable': is_correct_handling,
                    'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
                }
                results.append(result)
                duration_str = f"{duration:.2f}s" if duration is not None else "N/A"
                confidence_str = f"{result['confidence_if_any']:.2f}%" if result['confidence_if_any'] is not None else "N/A"
                server_duration_str = f"{server_duration_s:.4f}s" if server_duration_s is not None else "N/A"
                handling_status = "Correctly Handled (Unreliable)" if is_correct_handling else "Incorrectly Handled (Reliable)"
                if is_reliable_response is None and not result['error']:
                    handling_status = "Unknown (Reliable flag missing)"
                elif result['error']:
                    handling_status = f"Error: {result['error']}"

                print(f"[TC-008] {audio_file.name} (Type: {source_description}): Handling={handling_status}, " \
                      f"Pred={result['predicted_reciter_if_any']}, Conf={confidence_str}, " \
                      f"ServerResponseTime={server_duration_str}, ClientTime={elapsed:.2f}s, Status={status_code}")
    
    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-008.csv'
    json_path = report_dir / 'results_tc-008.json'
    summary_txt_path = report_dir / 'summary_tc-008.txt'
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['file', 'duration_seconds', 'source_description', 'server_reliable_flag', 
                      'predicted_reciter_if_any', 'confidence_if_any', 'correctly_handled_as_unreliable', 
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-008] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-008 (similar to TC-002) --- 
    total_files = len(results)
    correctly_handled_count = sum(1 for r in results if r['correctly_handled_as_unreliable'])
    false_positive_identification_count = total_files - correctly_handled_count
    
    handling_accuracy = (correctly_handled_count / total_files) * 100 if total_files > 0 else 0
    false_positive_rate = (false_positive_identification_count / total_files) * 100 if total_files > 0 else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    error_files_tc008 = [r['file'] for r in results if r['error']]

    conf_false_positives = [r['confidence_if_any'] for r in results if r['server_reliable_flag'] is True and r['confidence_if_any'] is not None]
    avg_conf_false_positive = (sum(conf_false_positives) / len(conf_false_positives)) if conf_false_positives else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-008 Summary: Reciter ID with Silence or Non-Speech Audio\n")
        sf_txt.write(f"------------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Processed: {total_files}\n")
        sf_txt.write(f"Correctly Handled (Marked Unreliable): {correctly_handled_count} ({handling_accuracy:.2f}%)\n")
        sf_txt.write(f"Incorrectly Handled (Marked Reliable - False Positive ID): {false_positive_identification_count} ({false_positive_rate:.2f}%)\n")
        sf_txt.write(f"Avg. Confidence of False Positive IDs: {avg_conf_false_positive:.2f}%\n" if conf_false_positives else "Avg. Confidence of False Positive IDs: N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time: {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if error_files_tc008:
            sf_txt.write(f"Files with Errors/Failures ({len(error_files_tc008)}):\n")
            for err_file in error_files_tc008:
                sf_txt.write(f"  - {err_file}\n")
        else:
            sf_txt.write("No errors or failures reported during processing of silence/non-speech audio.\n")
    
    print(f"[TC-008] TXT Summary saved: {summary_txt_path}")

def run_tc_009(endpoint_url, report_dir, model_info=None):
    """Run TC-009: Ayah Identification with very short audio clips."""
    tc_name = 'tc-009'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-009.")
        return

    audio_files = []
    for f in tc_path.iterdir():
        if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS:
            if len(f.stem) >= 6 and f.stem[:6].isdigit():
                audio_files.append(f)
            else:
                print(f"[TC-009] Warning: File {f.name} does not match SSSAAA naming pattern prefix. Skipping.")
    
    results = []
    print(f"\nRunning Test Case: {tc_name} - Ayah Identification with very short audio clips")

    for audio_file in audio_files:
        filename_stem_prefix = audio_file.stem[:6]
        try:
            expected_surah = int(filename_stem_prefix[:3])
            expected_ayah = int(filename_stem_prefix[3:])
        except ValueError:
            print(f"[TC-009] Warning: Could not parse Surah/Ayah from filename {audio_file.name}. Skipping.")
            continue

        duration = None
        try:
            duration = sf.info(str(audio_file)).duration
            if duration > 3.5: # Example threshold, can be adjusted
                 print(f"[TC-009] Warning: File {audio_file.name} is {duration:.2f}s, may not be a 'very short' Ayah clip. Proceeding anyway.")
        except Exception as e:
            print(f"Warning: Could not get duration for {audio_file.name}: {e}")

        with open(audio_file, 'rb') as f_audio:
            files = {'audio': (audio_file.name, f_audio, 'audio/mpeg')}
            payload = {} # Use server defaults for Ayah ID params
            start_time = time.time()
            try:
                response = requests.post(endpoint_url + '/getAyah', files=files, data=payload)
                elapsed = time.time() - start_time
                status_code = response.status_code
                try:
                    resp_json = response.json()
                except Exception:
                    resp_json = {'error': 'Invalid JSON response'}
            except Exception as e:
                elapsed = time.time() - start_time
                status_code = None
                resp_json = {'error': str(e)}
            
            server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
            server_duration_s = None
            if server_response_time_ms is not None:
                server_duration_s = server_response_time_ms / 1000.0

            best_match_data = resp_json.get('best_match') if isinstance(resp_json, dict) else None
            matches_list = resp_json.get('matches', []) if isinstance(resp_json, dict) else []
            transcription = resp_json.get('transcription', '') if isinstance(resp_json, dict) else resp_json.get('asr_transcription', '')
            
            top_match_correct = False
            found_in_matches = False
            top_match_details = {}

            if best_match_data:
                top_match_details = {
                    'surah': best_match_data.get('surah_number_en'),
                    'ayah': best_match_data.get('ayah_number_en'),
                    'text': best_match_data.get('ayah_text'),
                    'confidence': best_match_data.get('confidence_score')
                }
                if top_match_details['surah'] == expected_surah and top_match_details['ayah'] == expected_ayah:
                    top_match_correct = True
            
            for match_item in matches_list:
                if match_item.get('surah_number_en') == expected_surah and match_item.get('ayah_number_en') == expected_ayah:
                    found_in_matches = True
                    break
            
            result = {
                'file': str(audio_file),
                'duration_seconds': duration,
                'expected_surah': expected_surah,
                'expected_ayah': expected_ayah,
                'status_code': status_code,
                'client_round_trip_time_s': elapsed,
                'server_response_time_s': server_duration_s,
                'transcription': transcription,
                'top_match_surah': top_match_details.get('surah'),
                'top_match_ayah': top_match_details.get('ayah'),
                'top_match_text': top_match_details.get('text'),
                'top_match_confidence': top_match_details.get('confidence'),
                'top_match_correct': top_match_correct,
                'found_in_matches': found_in_matches,
                'num_matches_returned': len(matches_list),
                'all_matches': matches_list,
                'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
            }
            results.append(result)

            status_msg = "Correct (Top)" if top_match_correct else (
                "Correct (In List)" if found_in_matches else "Incorrect"
            )
            if result['error']:
                status_msg = f"Error: {result['error']}"
            elif not matches_list and not result['error']:
                status_msg = "No Matches Returned"

            conf_str = f"{top_match_details.get('confidence'):.2f}" if top_match_details.get('confidence') is not None else "N/A"
            print(f"[TC-009] {audio_file.name} (Exp: {expected_surah}:{expected_ayah}): Status={status_msg}, " \
                  f"TopMatch=({top_match_details.get('surah', '-')}:{top_match_details.get('ayah', '-')}, Conf={conf_str}), " \
                  f"Matches Returned={len(matches_list)}, ServerTime={server_duration_s if server_duration_s is not None else 'N/A'}")

    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-009.csv'
    json_path = report_dir / 'results_tc-009.json'
    summary_txt_path = report_dir / 'summary_tc-009.txt'

    csv_fieldnames = ['file', 'duration_seconds', 'expected_surah', 'expected_ayah', 
                      'top_match_surah', 'top_match_ayah', 'top_match_text', 'top_match_confidence', 
                      'top_match_correct', 'found_in_matches', 'num_matches_returned', 'transcription',
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in csv_fieldnames})
    
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-009] Reports saved: {csv_path}, {json_path}")

    total_files = len(results)
    errors_count_tc009 = sum(1 for r in results if r['error'])
    files_processed_successfully = total_files - errors_count_tc009
    
    top_match_correct_count = sum(1 for r in results if r['top_match_correct'])
    found_in_matches_count = sum(1 for r in results if r['found_in_matches'] and not r['top_match_correct'])
    
    accuracy_top_match = (top_match_correct_count / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0
    accuracy_found_in_list = ((top_match_correct_count + found_in_matches_count) / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0

    conf_correct_top = [r['top_match_confidence'] for r in results if r['top_match_correct'] and r['top_match_confidence'] is not None]
    avg_conf_correct_top = (sum(conf_correct_top) / len(conf_correct_top)) if conf_correct_top else 0

    conf_incorrect_top_but_match_returned = [
        r['top_match_confidence'] for r in results 
        if not r['top_match_correct'] and r['num_matches_returned'] > 0 and r['top_match_confidence'] is not None
    ]
    avg_conf_incorrect_top = (sum(conf_incorrect_top_but_match_returned) / len(conf_incorrect_top_but_match_returned)) if conf_incorrect_top_but_match_returned else 0

    num_matches_returned_list = [r['num_matches_returned'] for r in results if not r['error']]
    avg_matches_returned = (sum(num_matches_returned_list) / len(num_matches_returned_list)) if num_matches_returned_list else 0

    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None and not r['error']]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-009 Summary: Ayah Identification with Very Short Audio Clips\n")
        sf_txt.write(f"---------------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Submitted: {total_files}\n")
        sf_txt.write(f"Files Processed Successfully (no errors): {files_processed_successfully}\n")
        sf_txt.write(f"Files with Errors: {errors_count_tc009}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Accuracy (Top Match Correct): {top_match_correct_count}/{files_processed_successfully} ({accuracy_top_match:.2f}%)\n")
        sf_txt.write(f"Accuracy (Found in Returned List, incl. top): {(top_match_correct_count + found_in_matches_count)}/{files_processed_successfully} ({accuracy_found_in_list:.2f}%)\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Confidence (Correct Top Match): {avg_conf_correct_top:.2f}\n" if conf_correct_top else "Avg. Confidence (Correct Top Match): N/A\n")
        sf_txt.write(f"Avg. Confidence (Incorrect Top Match, but matches returned): {avg_conf_incorrect_top:.2f}\n" if conf_incorrect_top_but_match_returned else "Avg. Confidence (Incorrect Top Match): N/A\n")
        sf_txt.write(f"Avg. Number of Matches Returned per File: {avg_matches_returned:.2f}\n" if num_matches_returned_list else "Avg. Number of Matches Returned per File: N/A\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time (Successful Files): {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if errors_count_tc009 > 0:
            sf_txt.write(f"Files with Errors Details ({errors_count_tc009}):\n")
            for r_err in results:
                if r_err['error']:
                    sf_txt.write(f"  - {r_err['file']}: {r_err['error']}\n")
        else:
            sf_txt.write("No errors reported during Ayah identification processing with very short clips.\n")

    print(f"[TC-009] TXT Summary saved: {summary_txt_path}")

def run_tc_010(endpoint_url, report_dir, model_info=None):
    """Run TC-010: Ayah ID with silence or non-speech audio."""
    tc_name = 'tc-010'
    tc_path = Path('data/test-cases') / tc_name
    if not tc_path.exists():
        print(f"Test case folder {tc_path} does not exist. Skipping TC-010.")
        return
    
    # In TC-010, subdirectories categorize non-speech types (e.g., "Silence", "Office_Noise")
    sound_type_dirs = [d for d in tc_path.iterdir() if d.is_dir()]
    results = []

    print(f"\nRunning Test Case: {tc_name} - Ayah ID with silence or non-speech audio")

    for sound_dir in sound_type_dirs:
        source_description = sound_dir.name 
        audio_files = [f for f in sound_dir.iterdir() if f.is_file() and f.suffix.lower() in SUPPORTED_AUDIO_EXTS]
        
        for audio_file in audio_files:
            duration = None
            try:
                duration = sf.info(str(audio_file)).duration
            except Exception as e:
                print(f"Warning: Could not get duration for {audio_file.name}: {e}")

            with open(audio_file, 'rb') as f_audio:
                files = {'audio': (audio_file.name, f_audio, 'audio/mpeg')}
                payload = {} # Use server defaults for Ayah ID params
                start_time = time.time()
                try:
                    response = requests.post(endpoint_url + '/getAyah', files=files, data=payload)
                    elapsed = time.time() - start_time
                    status_code = response.status_code
                    try:
                        resp_json = response.json()
                    except Exception:
                        resp_json = {'error': 'Invalid JSON response'}
                except Exception as e:
                    elapsed = time.time() - start_time
                    status_code = None
                    resp_json = {'error': str(e)}
                
                server_response_time_ms = resp_json.get('response_time_ms') if isinstance(resp_json, dict) else None
                server_duration_s = None
                if server_response_time_ms is not None:
                    server_duration_s = server_response_time_ms / 1000.0

                matches_list = resp_json.get('matches', []) if isinstance(resp_json, dict) else []
                transcription = resp_json.get('transcription', '') if isinstance(resp_json, dict) else resp_json.get('asr_transcription', '')
                
                # For TC-010 (silence/non-speech), correct handling is NO matches found.
                # Check if 'matches_found' field exists, otherwise derive from matches_list length
                matches_found_server_flag = resp_json.get('matches_found') # Get the server's flag if present
                if matches_found_server_flag is None: # If server doesn't provide the flag explicitly
                    matches_found_server_flag = len(matches_list) > 0 # Derive it
                
                correctly_handled_no_match = not matches_found_server_flag # Correct if no matches were found

                result = {
                    'file': str(audio_file),
                    'duration_seconds': duration,
                    'source_description': source_description,
                    'status_code': status_code,
                    'client_round_trip_time_s': elapsed,
                    'server_response_time_s': server_duration_s,
                    'transcription': transcription,
                    'matches_found_server_flag': matches_found_server_flag, 
                    'num_matches_returned': len(matches_list),
                    'correctly_handled_no_match': correctly_handled_no_match,
                    'all_matches': matches_list, # Store for detailed JSON report
                    'error': resp_json.get('error') if isinstance(resp_json, dict) and 'error' in resp_json else '',
                }
                results.append(result)

                handling_status = "Correctly Handled (No Matches)" if correctly_handled_no_match else f"Incorrectly Handled (Matches Found: {len(matches_list)})"
                if result['error']:
                    handling_status = f"Error: {result['error']}"
                
                print(f"[TC-010] {audio_file.name} (Type: {source_description}): Handling={handling_status}, " \
                      f"Transcription='{transcription[:50]}...', ServerTime={server_duration_s if server_duration_s is not None else 'N/A'}")

    ensure_dir(report_dir)
    csv_path = report_dir / 'results_tc-010.csv'
    json_path = report_dir / 'results_tc-010.json'
    summary_txt_path = report_dir / 'summary_tc-010.txt'

    # For CSV, exclude 'all_matches'
    csv_fieldnames = ['file', 'duration_seconds', 'source_description', 'correctly_handled_no_match',
                      'num_matches_returned', 'matches_found_server_flag', 'transcription',
                      'client_round_trip_time_s', 'server_response_time_s', 'status_code', 'error']
    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, '') for k in csv_fieldnames})
    
    # Full data goes to JSON
    report_data_for_json = {'results': results}
    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(report_data_for_json, jf, ensure_ascii=False, indent=2)
    print(f"[TC-010] Reports saved: {csv_path}, {json_path}")

    # --- Generate TXT Summary for TC-010 --- 
    total_files = len(results)
    errors_count_tc010 = sum(1 for r in results if r['error'])
    files_processed_successfully = total_files - errors_count_tc010

    correctly_ignored_count = sum(1 for r in results if r['correctly_handled_no_match'] and not r['error'])
    false_positive_match_count = files_processed_successfully - correctly_ignored_count

    accuracy_no_match = (correctly_ignored_count / files_processed_successfully) * 100 if files_processed_successfully > 0 else 0
    
    server_times = [r['server_response_time_s'] for r in results if r['server_response_time_s'] is not None and not r['error']]
    avg_server_time_s = (sum(server_times) / len(server_times)) if server_times else 0

    with open(summary_txt_path, 'w', encoding='utf-8') as sf_txt:
        sf_txt.write(f"Test Case TC-010 Summary: Ayah ID with Silence or Non-Speech Audio\n")
        sf_txt.write(f"----------------------------------------------------------------------\n")
        sf_txt.write(f"Total Audio Files Submitted: {total_files}\n")
        sf_txt.write(f"Files Processed Successfully (no errors): {files_processed_successfully}\n")
        sf_txt.write(f"Files with Errors: {errors_count_tc010}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Correctly Handled (No Matches Returned for Non-Speech): {correctly_ignored_count}/{files_processed_successfully} ({accuracy_no_match:.2f}%)\n")
        sf_txt.write(f"Incorrectly Handled (Matches Returned for Non-Speech - False Positives): {false_positive_match_count}/{files_processed_successfully}\n")
        sf_txt.write(f"\n")
        sf_txt.write(f"Avg. Server Response Time (Successful Files): {avg_server_time_s:.4f}s\n" if server_times else "Avg. Server Response Time: N/A\n")
        sf_txt.write(f"\n")
        if errors_count_tc010 > 0:
            sf_txt.write(f"Files with Errors Details ({errors_count_tc010}):\n")
            for r_err in results:
                if r_err['error']:
                    sf_txt.write(f"  - {r_err['file']}: {r_err['error']}\n")
        else:
            sf_txt.write("No errors reported during Ayah ID processing of silence/non-speech audio.\n")
        if false_positive_match_count > 0:
            sf_txt.write(f"\nFiles with False Positive Ayah Matches ({false_positive_match_count}):\n")
            for r_fp in results:
                if not r_fp['correctly_handled_no_match'] and not r_fp['error']:
                    sf_txt.write(f"  - {r_fp['file']} (Type: {r_fp['source_description']}) - Returned {r_fp['num_matches_returned']} matches\n")

    print(f"[TC-010] TXT Summary saved: {summary_txt_path}")

def main():
    print("\n🚀 Starting Test Case Runner...")
    parser = argparse.ArgumentParser(description="Unified Test Case Runner for Qurra Backend.")
    parser.add_argument('--test-case', type=str, help='Test case to run (e.g., tc-001). If omitted, runs all available test cases.')
    parser.add_argument('--endpoint', type=str, default='http://localhost:5000', help='Base URL for backend server.')
    args = parser.parse_args()

    print(f"ℹ️ Endpoint: {args.endpoint}")

    if not check_server_available(args.endpoint):
        print(f"🛑 ERROR: Server is not running or not reachable at {args.endpoint}. Please start the server and try again.")
        return

    run_id = generate_run_id()
    base_report_dir = Path('test-cases-report') / run_id
    ensure_dir(base_report_dir)
    print(f"📂 Report directory: {base_report_dir.resolve()}")

    model_info = get_model_info(args.endpoint)
    if model_info:
        model_info_path = base_report_dir / 'model_info.json'
        with open(model_info_path, 'w', encoding='utf-8') as jf:
            json.dump(model_info, jf, ensure_ascii=False, indent=2)
        print(f"✓ Model info saved to {model_info_path.name}")
    else:
        print(f"⚠️ Could not retrieve model info. Proceeding without it.")


    available_cases_on_disk = get_test_cases()
    # Filter available_cases_on_disk against TC_DESCRIPTIONS to only run known/implemented TCs
    # TC_DESCRIPTIONS now serves as the master list of runnable and defined test cases.
    runnable_tcs_defined = {tc: desc for tc, desc in TC_DESCRIPTIONS.items() if tc in available_cases_on_disk}
    
    if not runnable_tcs_defined:
        print(f"🛑 No runnable test cases found in 'data/test-cases' that are also defined in TC_DESCRIPTIONS. Existing folders on disk: {available_cases_on_disk}")
        return

    cases_to_run_names = []
    if args.test_case:
        if args.test_case in runnable_tcs_defined:
            cases_to_run_names = [args.test_case]
            print(f"ℹ️ Running specified test case: {args.test_case} - {runnable_tcs_defined[args.test_case]}")
        else:
            print(f"🛑 Specified test case '{args.test_case}' is not available on disk or not defined in TC_DESCRIPTIONS. Available defined: {list(runnable_tcs_defined.keys())}")
            return
    else:
        # TC_DESCRIPTIONS is the source of truth for what is runnable.
        # We sort the keys from runnable_tcs_defined to ensure a consistent run order.
        cases_to_run_names = sorted(list(runnable_tcs_defined.keys())) 
        if not cases_to_run_names:
            # This state should ideally be caught by the earlier check on runnable_tcs_defined
            print(f"ℹ️ No test cases to run. Check 'data/test-cases' and TC_DESCRIPTIONS in the script.")
            return
        print(f"ℹ️ Running all available and defined test cases: {cases_to_run_names}")


    for tc_name in cases_to_run_names:
        report_dir = base_report_dir / tc_name
        tc_description = runnable_tcs_defined[tc_name]
        print(f"\n🧪 Running Test Case: {tc_name} - {tc_description}")
        
        # Create specific subdirectory for this TC's reports
        ensure_dir(report_dir) 

        if tc_name == 'tc-001':
            run_tc_001(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-002':
            run_tc_002(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-003':
            run_tc_003(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-004':
            run_tc_004(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-005':
            run_tc_005(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-006':
            run_tc_006(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-007':
            run_tc_007(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-008':
            run_tc_008(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-009':
            run_tc_009(args.endpoint, report_dir, model_info=model_info)
        elif tc_name == 'tc-010':
            run_tc_010(args.endpoint, report_dir, model_info=model_info)
        else:
            print(f"🤷 Test case {tc_name} is defined but not implemented in the main runner loop.")

    print(f"\n✨ All selected test cases completed. Reports in {base_report_dir.resolve()}")

if __name__ == "__main__":
    main() 