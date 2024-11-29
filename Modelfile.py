

# Créer un new modelfile
import ollama


modelfile = """
FROM llama3.2
SYSTEM You are a very smart assistant who knows everything about oceans. You are very succinte
PARAMETER temperature 0.1
"""

ollama.create(model="knowitall", modelfile=modelfile)

res = ollama.generate(model="knowitall", prompt="why the ocean is so salty ?")
print(res['response'])