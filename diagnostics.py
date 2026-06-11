from openai import OpenAI
from dotenv import load_dotenv
import os
import json

T, F = True, False

class Diagnostics:

    def __init__(self):
        load_dotenv()
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://ellm.nrp-nautilus.io/v1"
        )
        
    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):

        prompt = f"""
        You are solving a Bayesian network diagnosis problem.

        Use this Bayesian network:

        P(asia=T) = 0.01
        P(smoking=T) = 0.50

        P(tuberculosis=T | asia=T) = 0.05
        P(tuberculosis=T | asia=F) = 0.01

        P(cancer=T | smoking=T) = 0.10
        P(cancer=T | smoking=F) = 0.01

        P(bronchitis=T | smoking=T) = 0.60
        P(bronchitis=T | smoking=F) = 0.30

        either = tuberculosis OR cancer

        P(xray=abnormal | either=T) = 0.99
        P(xray=abnormal | either=F) = 0.05

        P(dyspnea=present | bronchitis=T, either=T) = 0.90
        P(dyspnea=present | bronchitis=T, either=F) = 0.80
        P(dyspnea=present | bronchitis=F, either=T) = 0.70
        P(dyspnea=present | bronchitis=F, either=F) = 0.10

        Evidence:
        visit_to_asia={visit_to_asia}
        smoking={smoking}
        xray_result={xray_result}
        dyspnea={dyspnea}

        Compute the posterior probability of each disease (TB, Cancer, Bronchitis) given
        the evidence above. Return the disease with the highest posterior probability
        and its probability value.
        """

        response = self.client.chat.completions.create(
            model="gpt-oss",
            messages=[{"role": "user", "content": prompt}],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "diagnosis_result",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {
                            "disease": {
                                "type": "string",
                                "enum": ["TB", "Cancer", "Bronchitis"]
                            },
                            "probability": {
                                "type": "number"
                            }
                        },
                        "required": ["disease", "probability"],
                        "additionalProperties": False
                    }
                }
            }
        )

        result = json.loads(response.choices[0].message.content)

        return [result["disease"], result["probability"]]
