# Import BayesNet for building the Bayesian network and enumeration_ask for exact inference
from probability4e import BayesNet, enumeration_ask

# Diagnostics class encapsulates the Asia Bayesian network and exposes a diagnose method
class Diagnostics:

    def __init__(self):
        # Build the Bayesian network using the classic "Asia" network structure.
        # Each tuple is (node_name, parent_names, CPT) where CPT is:
        #   - a float for root nodes (prior probability of being True)
        #   - a dict mapping parent truth-value(s) to P(node=True | parents)
        self.net = BayesNet([
            # Root node: prior probability of 1% that the patient visited Asia
            ('asia', '', 0.01),
            # Root node: prior probability of 50% that the patient is a smoker
            ('smoking', '', 0.5),
            # Tuberculosis depends on Asia visit: 5% chance if visited, 1% if not
            ('tuberculosis', 'asia', {True: 0.05, False: 0.01}),
            # Lung cancer depends on smoking: 10% chance if smoker, 1% if not
            ('lung_cancer', 'smoking', {True: 0.1, False: 0.01}),
            # Bronchitis depends on smoking: 60% chance if smoker, 30% if not
            ('bronchitis', 'smoking', {True: 0.6, False: 0.3}),
            # 'either' is True if tuberculosis OR lung cancer is present (logical OR gate):
            # True whenever at least one of the two parents is True, False only if both are False
            ('either', 'tuberculosis lung_cancer', {(True, True): 1.0, (True, False): 1.0, (False, True): 1.0, (False, False): 0.0}),
            # X-ray result depends on 'either': 99% abnormal if either disease present, 5% false-positive if not
            ('xray', 'either', {True: 0.99, False: 0.05}),
            # Dyspnea (shortness of breath) depends on both 'either' and bronchitis
            # Tuple keys are (either, bronchitis); probability of dyspnea given each combination
            ('dyspnea', 'either bronchitis', {(True, True): 0.9, (True, False): 0.7, (False, True): 0.8, (False, False): 0.1})
        ])

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):
        # Build the evidence dictionary from the user-provided symptom/observation strings

        evidence = {}

        # Map the Asia visit answer to a boolean and add it to evidence
        if visit_to_asia.lower() == 'yes':
            evidence['asia'] = True
        elif visit_to_asia.lower() == 'no':
            evidence['asia'] = False

        # Map the smoking answer to a boolean and add it to evidence
        if smoking.lower() == 'yes':
            evidence['smoking'] = True
        elif smoking.lower() == 'no':
            evidence['smoking'] = False

        # Map the X-ray result to a boolean (abnormal = True) and add it to evidence
        if xray_result.lower() == 'abnormal':
            evidence['xray'] = True
        elif xray_result.lower() == 'normal':
            evidence['xray'] = False

        # Map the dyspnea symptom to a boolean (present = True) and add it to evidence
        if dyspnea.lower() == 'present':
            evidence['dyspnea'] = True
        elif dyspnea.lower() == 'absent':
            evidence['dyspnea'] = False

        # Use exact inference (enumeration) to compute P(tuberculosis=True | evidence)
        tb_prob = enumeration_ask('tuberculosis', evidence, self.net)[True]
        # Use exact inference to compute P(lung_cancer=True | evidence)
        cancer_prob = enumeration_ask('lung_cancer', evidence, self.net)[True]
        # Use exact inference to compute P(bronchitis=True | evidence)
        bronchitis_prob = enumeration_ask('bronchitis', evidence, self.net)[True]

        # Collect the three posterior probabilities into a labelled dictionary
        probabilities = {
            "TB": tb_prob,
            "Lung Cancer": cancer_prob,
            "Bronchitis": bronchitis_prob
        }

        # Find the disease with the highest posterior probability
        disease = max(probabilities, key=probabilities.get)

        # Return the most likely disease and its probability as a two-element list
        return [disease, probabilities[disease]]

