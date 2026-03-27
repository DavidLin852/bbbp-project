"""
Test script for BBB transport mechanism prediction.
"""

from src.path_prediction.mechanism_predictor_cornelissen import MechanismPredictor

# Initialize predictor
predictor = MechanismPredictor()

# Example molecules with diverse properties
test_molecules = [
    ('Aspirin', 'CC(=O)OC1=CC=CC=C1C(=O)O'),
    ('Dopamine', 'NCCc1cc(O)c(O)cc1'),
    ('Caffeine', 'Cn1cnc2c1c(=O)n(C)c(=O)n2C'),
    ('Glucose', 'OCC1OC(O)C(O)C(O)C1O'),
    ('Nicotine', 'CN1CCCC1c1cccnc1'),
    ('Morphine', 'CN1CCC23C4C1CC5=C2C(=C(C=C5)O)OCC3C4'),
]

print('='*100)
print('BBB Transport Mechanism Prediction (based on Cornelissen et al. 2022)')
print('='*100)
print()

for name, smiles in test_molecules:
    print(f'{name:^95}')
    print(f'SMILES: {smiles}')
    print()

    results = predictor.predict_all(smiles)

    # Format results
    for mech, res in results.items():
        if 'error' not in res:
            pred_str = '[+YES]' if res['prediction'] else '[-NO]'
            prob_str = f"{res['probability']:.1%}"
            conf_str = res['confidence']

            # Add interpretation
            if mech == 'BBB':
                interpretation = 'Can cross BBB' if res['prediction'] else 'Cannot cross BBB'
            elif mech == 'Influx':
                interpretation = 'Active transport into brain' if res['prediction'] else 'No active influx'
            elif mech == 'Efflux':
                interpretation = 'Efflux pump substrate (P-gp)' if res['prediction'] else 'Not efflux substrate'
            elif mech == 'PAMPA':
                interpretation = 'Strong passive diffusion' if res['prediction'] else 'Weak passive diffusion'
            elif mech == 'CNS':
                interpretation = 'Has CNS activity' if res['prediction'] else 'No CNS activity'
            else:
                interpretation = ''

            print(f'  {mech:6s}: {pred_str:8} | Prob: {prob_str:6} | Conf: {conf_str:6} | {interpretation}')
        else:
            print(f'  {mech:6s}: Error - {res["error"]!r}')

    print()
    print('-'*100)
    print()
