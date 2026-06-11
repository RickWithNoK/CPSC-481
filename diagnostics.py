from openai import OpenAI
from dotenv import load_dotenv
import os
import json

T, F = True, False

# Diagnostics class uses an LLM to solve the Asia Bayesian network diagnosis problem
class Diagnostics:

    def __init__(self):
        # Load environment variables from .env file (API key is stored there, not in code)
        load_dotenv()
        # Initialize the OpenAI client with the API key from the environment
        self.client = OpenAI(
            api_key=os.getenv("API_KEY"),
            base_url="https://ellm.nrp-nautilus.io/v1"
        )

    def diagnose(self, visit_to_asia, smoking, xray_result, dyspnea):

        # Print the received evidence before building the prompt
        print("Evidence Received")
        print(f"Visit to Asia : {visit_to_asia}")
        print(f"Smoking       : {smoking}")
        print(f"X-ray result  : {xray_result}")
        print(f"Dyspnea       : {dyspnea}")

        # Build the prompt with the full Bayesian network structure and the observed evidence
        print("Building Prompt")
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
        print("\nPrompt built successfully.")

        # Send the prompt to the LLM and request structured JSON output
        print("\nSending Request to LLM")
        response = self.client.chat.completions.create(
            model="gpt-oss",
            messages=[{"role": "user", "content": prompt}],
            # Structured output schema enforces the exact response format
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
        print("\nResponse received.")

        # Parse the structured JSON response from the LLM
        print("Parsing LLM Response")
        raw = response.choices[0].message.content
        print(f"Raw response: {raw}")
        result = json.loads(raw)

        # Extract the most likely disease and its posterior probability
        disease = result["disease"]
        probability = result["probability"]
        print("\nDiagnosis Result")
        print(f"Most likely disease : {disease}")
        print(f"Probability         : {probability:.4f}")

        return [disease, probability]
