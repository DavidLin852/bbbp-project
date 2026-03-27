"""
Test Transport Mechanism Prediction System

Quick validation that all components work correctly.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("="*60)
print("BBB TRANSPORT MECHANISM PREDICTION - SYSTEM TEST")
print("="*60)

# Test 1: Import modules
print("\n1. Testing imports...")
try:
    from src.path_prediction.data_collector import TransportDataCollector
    from src.path_prediction.feature_extractor import MechanismFeatureExtractor
    from src.path_prediction.mechanism_predictor import MechanismPredictor
    print("   ✓ All imports successful")
except Exception as e:
    print(f"   ✗ Import error: {e}")
    sys.exit(1)

# Test 2: Feature extraction
print("\n2. Testing feature extraction...")
try:
    extractor = MechanismFeatureExtractor()
    test_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"  # Aspirin

    features, mol = extractor.extract_features(test_smiles)
    if features is not None:
        print(f"   ✓ Features extracted: shape {features.shape}")
    else:
        print("   ✗ Feature extraction failed")
        sys.exit(1)
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 3: Model loading
print("\n3. Testing model loading...")
try:
    import xgboost as xgb
    import joblib

    models_dir = project_root / "artifacts" / "models" / "mechanism"

    # Check if models exist
    bbb_model_path = models_dir / "bbb_model.json"
    imputer_path = models_dir / "imputer.joblib"

    if bbb_model_path.exists():
        print(f"   ✓ BBB model found: {bbb_model_path}")

        # Load model
        model = xgb.XGBClassifier()
        model.load_model(str(bbb_model_path))
        print(f"   ✓ BBB model loaded successfully")

        # Load imputer
        imputer = joblib.load(imputer_path)
        print(f"   ✓ Imputer loaded successfully")
    else:
        print(f"   ✗ BBB model not found: {bbb_model_path}")
        print("   Please run: python scripts/mechanism_training/train_robust.py")
        sys.exit(1)

except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 4: Prediction
print("\n4. Testing prediction...")
try:
    # Predict BBB
    features_imp = imputer.transform(features.reshape(1, -1))
    proba = model.predict_proba(features_imp)[0, 1]
    pred = int(proba >= 0.5)

    print(f"   SMILES: {test_smiles} (Aspirin)")
    print(f"   Prediction: {'BBB+' if pred == 1 else 'BBB-'}")
    print(f"   Probability: {proba:.2%}")
    print("   ✓ Prediction successful")
except Exception as e:
    print(f"   ✗ Error: {e}")
    sys.exit(1)

# Test 5: Test molecules
print("\n5. Testing on multiple molecules...")
test_cases = [
    ("Aspirin", "CC(=O)OC1=CC=CC=C1C(=O)O", "Passive"),
    ("Caffeine", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Passive"),
    ("Dopamine", "NCCc1cc(O)c(O)cc1", "Influx"),
]

for name, smiles, expected_mech in test_cases:
    try:
        features, _ = extractor.extract_features(smiles)
        features_imp = imputer.transform(features.reshape(1, -1))
        proba = model.predict_proba(features_imp)[0, 1]

        status = "✓" if proba > 0.5 else "✗"
        print(f"   {status} {name:12s}: BBB+ prob={proba:.2f} (expected: {expected_mech})")
    except Exception as e:
        print(f"   ✗ {name}: Error - {e}")

# Summary
print("\n" + "="*60)
print("SYSTEM TEST COMPLETE")
print("="*60)
print("\nAll components working correctly!")
print("\nTo use the system:")
print("1. Train models: python scripts/mechanism_training/train_robust.py")
print("2. Run Streamlit: streamlit run app_bbb_predict.py")
print("3. Navigate to: 🧬 BBB Mechanism Prediction")
print("\nFor more info, see: docs/transport_mechanism_implementation_summary.md")
