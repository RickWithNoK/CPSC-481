from probability4e import BayesNet, enumeration_ask

class Diagnostics:

    def __init__(self):
    
        self.net = BayesNet([ ('asia', '', 0.01),
                             ('smoking', '', 0.5), 
                             ('tuberculosis', 'asia', {True: 0.05, False: 0.01}),
                             ('lung_cancer', 'smoking', {True: 0.1, False: 0.01}),
                             ('bronchitis', 'smoking', {True: 0.6, False: 0.3}),
                             ('either', 'tuberculosis lung_cancer', {(True, True): 1.0, (True, False): 1.0, (False, True): 1.0, (False, False): 0.0}),
                             ('xray', 'either', {True: 0.99, False: 0.05}),
                             ('dyspnea', 'either bronchitis', {(True, True): 0.9, (True, False): 0.7, (False, True): 0.8, (False, False): 0.1})
                             ])

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):

        evidence = {}

        if visit_to_asia.lower() == 'yes':
            evidence['asia'] = True
        elif visit_to_asia.lower() == 'no':
            evidence['asia'] = False

        if smoking.lower() == 'yes':
            evidence['smoking'] = True
        elif smoking.lower() == 'no':
            evidence['smoking'] = False

        if xray_result.lower() == 'abnormal':
            evidence['xray'] = True
        elif xray_result.lower() == 'normal':
            evidence['xray'] = False
        
        if dyspnea.lower() == 'present':
            evidence['dyspnea'] = True 
        elif dyspnea.lower() == 'absent':
            evidence['dyspnea'] = False

        tb_prob = enumeration_ask('tuberculosis', evidence, self.net)[True]
        cancer_prob = enumeration_ask('lung_cancer', evidence, self.net)[True]
        bronchitis_prob = enumeration_ask('bronchitis', evidence, self.net)[True]

        probabilities = { "TB": tb_prob, "Lung Cancer": cancer_prob, "Bronchitis": bronchitis_prob 
                         }

        disease = max(probabilities, key=probabilities.get)

        return [disease, probabilities[disease]]

