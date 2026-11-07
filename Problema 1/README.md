![Grafica](Problema%201/curva_deteccion_gap_alg1.png)
esta curva dice qué tan separados tienen que estar los grupos para que el criterio los encuentre.


![Grafica](Problema%201/trampa_visual_tsne.png)
Esto confirma lo que advierte el enunciado: que t-SNE o UMAP muestren islas bien definidas no demuestra que existan grupos en los datos

![Grafica](Problema%201/gap_vinos.png)

El gap sube de forma clara al pasar de 1 a 3 grupos y después se estabiliza
Por eso el criterio selecciona k* = 3: agregar más grupos solo subdivide grupos que ya estaban bien definidos
Este resultado coincide con las tres variedades reales de vino
