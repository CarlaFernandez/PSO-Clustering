import random

words = ["caracola", "ventana", "pista",
         "violento", "domingo", "saltamontes", "tarjeta", "monitor", "teclado", "veinte"]

for i in range(10):
    string = ""
    for j in range(10):
        string += words[random.randint(0, len(words)-1)] + " "
    with open("/home/carla/Documentos/Multiagentes/implementation/PSO-Clustering/generated_documents/" + string.strip() + ".txt", "w") as f:
        f.write(string)