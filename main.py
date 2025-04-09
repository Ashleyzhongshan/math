
import numpy as np
import pandas as pd
import requests
A = [[6, -2, -1], [-2, 6, -1], [-1, -1, 5]]
# A = pd.DataFrame(np.array(A))

# Calculating the inverse of the matrix


from openai import OpenAI
api_key ="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"
client = OpenAI(api_key=api_key)

completion = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": "Tell me how to solve a polynomial equation."
        }
    ]
)

print(completion.choices[0].message)






if __name__ == '__main__':
   print("Hlelo")