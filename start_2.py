import ollama

response =ollama.list()

# print(response)
# ==Exemple de chat==
res =ollama.chat(
    model="llama3.2",
    messages=[
        {
            'role':"user", "content": "Hello, comment vas tu ?"
        }
    ]
)
#print(res) #ici toutes les info, model, payload, load_duration ...
print(res["message"]["content"]) #ici on obtient juste la rép

# ==Exemple straming==
# res =ollama.chat(
#     model="llama3.2",
#     messages=[
#         {
#             'role':"user", 
#             "content": "Why the ocean is so salty ?"
#         },
#     ],
#     stream=True,
# )
# for chunk in res:
#     print(chunk["message"]["content"], end="", flush=True) #pour voir le message s'afficher progessivement permet de récupérer la réponse par morceaux (chunks)

# # == Exemple Generate ==
# res = ollama.generate(
#     model="llama3.2",
#     prompt="Pourquoi le ciel est bleu ?",
# )
#show
# print(ollama.show("llama3.2"))



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

# delete model
# ollama.delete("knowitall")