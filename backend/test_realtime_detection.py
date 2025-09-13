#!/usr/bin/env python3
"""
Test script to verify real_time_detection parameter functionality.
Tests both real-time mode (ML only) and full analysis mode (ML + Gemini + Nano Banana).
"""

import requests
import json
import time
from pathlib import Path

# API endpoint
BASE_URL = "http://localhost:8000"
ANALYZE_ENDPOINT = f"{BASE_URL}/analyze"

def test_real_time_mode():
    """Test real-time detection mode (ML only, faster)"""
    print("\n=== Testing Real-Time Detection Mode ===")
    
    # Find a test image
    test_image_path = Path("../runs/car_defect_detection3").glob("*.jpg")
    test_image_path = next(test_image_path, None)
    
    if not test_image_path or not test_image_path.exists():
        print("No test image found. Creating a dummy request...")
        return
    
    try:
        start_time = time.time()
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            params = {'real_time_detection': True}
            response = requests.post(ANALYZE_ENDPOINT, files=files, params=params)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Real-time mode successful!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"🔍 Mode: {result.get('ai_analysis', {}).get('mode', 'unknown')}")
            print(f"🚗 Damage detected: {result.get('ai_analysis', {}).get('damage_detected', False)}")
            print(f"🖼️  AI repaired image: {result.get('ai_analysis', {}).get('ai_repaired_image') is not None}")
        else:
            print(f"❌ Real-time mode failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing real-time mode: {e}")

def test_full_analysis_mode():
    """Test full analysis mode (ML + Gemini + Nano Banana, slower but comprehensive)"""
    print("\n=== Testing Full Analysis Mode ===")
    
    # Find a test image
    test_image_path = Path("../runs/car_defect_detection3").glob("*.jpg")
    test_image_path = next(test_image_path, None)
    
    if not test_image_path or not test_image_path.exists():
        print("No test image found. Creating a dummy request...")
        return
    
    try:
        start_time = time.time()
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            params = {'real_time_detection': False}
            response = requests.post(ANALYZE_ENDPOINT, files=files, params=params)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Full analysis mode successful!")
            print(f"⏱️  Processing time: {processing_time:.2f} seconds")
            print(f"🔍 Mode: {result.get('ai_analysis', {}).get('mode', 'unknown')}")
            print(f"🚗 Damage detected: {result.get('ai_analysis', {}).get('damage_detected', False)}")
            print(f"🖼️  AI repaired image: {result.get('ai_analysis', {}).get('ai_repaired_image') is not None}")
        else:
            print(f"❌ Full analysis mode failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing full analysis mode: {e}")

def test_health_check():
    """Test if the API is running"""
    print("\n=== Testing API Health ===")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            print("✅ API is healthy and running")
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Starting Real-Time Detection Tests")
    
    # Check if API is running
    if not test_health_check():
        print("\n❌ API is not running. Please start the backend server first.")
        exit(1)
    
    # Test both modes
    test_real_time_mode()
    test_full_analysis_mode()
    
    print("\n🎉 Testing completed!")
    print("\n📊 Summary:")
    print("   • Real-time mode: Uses only ML model (faster)")
    print("   • Full analysis mode: Uses ML + Gemini + Nano Banana (slower, more comprehensive)")
    print("   • WebSocket real-time detection automatically uses real-time mode for speed")