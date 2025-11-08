"""Setup script for Feature 2: Carbon Intensity Service."""

import os

print("Setting up Feature 2: Carbon Intensity Service...")
print("=" * 60)

# Create providers directory
os.makedirs("app/providers", exist_ok=True)
print("✓ Created app/providers directory")

# The files are already created through the artifacts above
# This script just documents what needs to be done

print("""
📦 New Files Created:
  - app/providers/__init__.py
  - app/providers/intensity_base.py  
  - app/providers/simulator.py
  - app/providers/electricitymap.py
  - app/intensity.py

📝 Updated Files:
  - app/settings.py (added EM_* and cache settings)
  - app/main.py (added intensity router)
  - requirements.txt (added httpx, cachetools)
  - README.md (added carbon intensity docs)

🚀 Next Steps:

1. Install new dependencies:
   pip install httpx==0.27.2 cachetools==5.5.0

2. Restart the service:
   python -m app.main

3. Test simulator mode (no API key needed):
   python -c "import requests, json; r = requests.get('http://localhost:8000/intensity/current'); print(json.dumps(r.json(), indent=2))"

4. Test forecast:
   python -c "import requests, json; r = requests.get('http://localhost:8000/intensity/forecast'); print(json.dumps(r.json(), indent=2))"

5. (Optional) To use real ElectricityMap data:
   - Get API key from https://api-portal.electricitymap.org/
   - Set environment: export EM_API_KEY=your_key_here
   - Disable simulation: export ENABLE_SIMULATION=false
   - Restart service

""")

print("=" * 60)
print("✅ Feature 2 setup complete!")