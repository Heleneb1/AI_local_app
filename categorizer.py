import ollama
import os

import time

start_time = time.time()

model = "llama3.2"

input_file= "./data/grocery_list.txt"
output_file ="./data/categorized_grocery_list.txt"

#Si input file existe
if not os.path.exists(input_file):
    print(f"Input file `{input_file}`not found ")
    exit(1)

# LIre les articles non catégorisés
with open(input_file, "r", encoding="utf-8") as f: #rajouter utf-8 pour les caractères spéciaux
    items= f.read().strip()


# Préparer le prompt pour le model

prompt = f"""
You are an assistant that categorizes and sorts grocery items.

Here is a list of grocery items:

{items}

Please:

1. Categorize these items into appropriate categories such as 
Produits laitiers, viande, boulangerie, boissons, etc.
2. Sort the items alphabetically within each category.
3. Present the categorized list in a clear and organized manner, using bullet points or numbering.
4.Only include items in their correct categories. Do not mix items between categories.

"""
print("Lecture du fichier réussie...")
#Envoi du prompt et obtenir la réponse

try:
    response = ollama.generate(model=model, prompt=prompt)
    generated_text = response.get("response", "")
    print("==== Categorized List: ===== \n")
    print(generated_text)

    # Ecrire la liste catégorisée dans le fichier de sortie
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(generated_text.strip())

    print(f"Categorized grocery list has been saved to '{output_file}'.")
except Exception as e:
    print("An error occurred:", str(e))

end_time = time.time()
execution_time = end_time - start_time
print(f"Temps d'exécution total : {execution_time:.2f} secondes")