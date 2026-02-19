"""
Test client for the video prediction server
This script demonstrates how to use the API to upload videos and get predictions
"""

import requests
import json
import sys
import os

# Server URL
SERVER_URL = "http://localhost:5000"

def check_health():
    """Check if the server is running and healthy"""
    try:
        response = requests.get(f"{SERVER_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print("✓ Server is healthy")
            print(f"  Model loaded: {data['model_loaded']}")
            print(f"  Device: {data['device']}")
            return True
        else:
            print("✗ Server returned error:", response.status_code)
            return False
    except requests.exceptions.ConnectionError:
        print("✗ Cannot connect to server. Is it running?")
        return False

def get_model_info():
    """Get information about the model"""
    try:
        response = requests.get(f"{SERVER_URL}/info")
        if response.status_code == 200:
            data = response.json()
            model_info = data['model_info']
            print("\n=== Model Information ===")
            print(f"Action classes: {model_info['num_action_classes']}")
            print(f"App classes: {model_info['num_app_classes']}")
            print(f"Device: {model_info['device']}")
            print(f"\nAvailable actions: {', '.join(model_info['action_classes'][:10])}...")
            print(f"Available apps: {', '.join(model_info['app_classes'])}")
            return True
        else:
            print("Error getting model info:", response.status_code)
            return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def predict_video(video_path):
    """Upload a video and get predictions"""
    if not os.path.exists(video_path):
        print(f"✗ Video file not found: {video_path}")
        return None
    
    print(f"\n=== Predicting video: {os.path.basename(video_path)} ===")
    print("Uploading video...")
    
    try:
        with open(video_path, 'rb') as f:
            files = {'video': f}
            response = requests.post(f"{SERVER_URL}/predict", files=files)
        
        if response.status_code == 200:
            data = response.json()
            
            # Print video info
            video_info = data['video_info']
            print(f"\n✓ Video processed successfully!")
            print(f"  Original filename: {video_info['original_filename']}")
            print(f"  Duration: {video_info['duration_seconds']}s")
            print(f"  Chunks: {video_info['num_chunks']}")
            
            # Print summary
            summary = data['summary']
            print(f"\n=== Summary ===")
            print(f"  Most common action: {summary['most_common_action']}")
            print(f"  Most common app: {summary['most_common_app']}")
            print(f"  Avg action confidence: {summary['average_action_confidence']}")
            print(f"  Avg app confidence: {summary['average_app_confidence']}")
            
            # Print action distribution
            print(f"\n  Action distribution:")
            for action, count in summary['action_distribution'].items():
                print(f"    {action}: {count}")
            
            print(f"\n  App distribution:")
            for app, count in summary['app_distribution'].items():
                print(f"    {app}: {count}")
            
            # Print detailed predictions
            print(f"\n=== Detailed Predictions ===")
            for pred in data['predictions']:
                print(f"  [{pred['time_range']}] {pred['action']} ({pred['action_confidence']:.2f}) | {pred['app']} ({pred['app_confidence']:.2f})")
            
            # Save results to file
            output_file = f"predictions_{data['request_id']}.json"
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"\n✓ Full results saved to: {output_file}")
            
            return data
        else:
            error_data = response.json()
            print(f"✗ Prediction failed: {error_data.get('error', 'Unknown error')}")
            return None
    
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def main():
    print("=== Video Prediction Server Test Client ===\n")
    
    # Check server health
    if not check_health():
        print("\nPlease start the server first:")
        print("  python server/app.py")
        return
    
    # Get model info
    get_model_info()
    
    # Get video path from command line or use example
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
    else:
        print("\n=== Usage ===")
        print("  python server/test_client.py <video_path>")
        print("\nExample:")
        print("  python server/test_client.py data/videos_001/chunks/chunk_001.mp4")
        print("  python server/test_client.py my_video.mp4")
        return
    
    # Predict video
    predict_video(video_path)

if __name__ == '__main__':
    main()
