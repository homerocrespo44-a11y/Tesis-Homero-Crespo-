from matplotlib.lines import Line2D
import random

import numpy as np
import matplotlib.pyplot as plt


# Parámetros
n = 2000                   # Número de pasos
N = 100                   # Tamaño de la matriz
T = 400                   # Temperatura en K (afecta las probabilidades de reacción)
C = 5                     # Factor de escala (acelera la simulación aumentando las probabilidades, no afecta la física relativa entre procesos porque esta normalizado)
d = 2.77e-10              #distancia entre primeros vecinos de Pt (en metros, 2.77 Å)
epsilon = 8.854*10**-12   #permitividad del vacio (en C^2/(N*m^2)), probar tambien con la del agua, Mati me dijo que use la del agua pero el en su papper tiene la del vacio
z = 1.5*10**-10           #distancia entre membrana y pt (valor inventado, charlarlo con mati)


Eb = {                    # Barreras (en eV)
    "H": 0.003,           # Energía de activación para mover H+, inventada (en eV)
    "O": 0.41,            # Energía de activación para mover O (en eV)
    "OH": 0.47,           # Energía de activación para mover OH (en eV)
    "O2": 0.31,           # Energía de activación para mover O2 (en eV)
    "H2O2": 0.21,         # Energía de activación para mover H2O2 (en eV)
    "OOH": 0.21           # Energía de activación para mover OOH (en eV)
}

# Barreras para procesos donde las partículas se unen o cambian de estado químico
Ec = {
    "H + O -> OH": 0.912,   #valor de paper es 0.912 lo cambie para que se vea algo de agua en la simulación, sino no se formaba nada
    "OH + H -> H2O": 0.197,
    "O + O -> O2": 1.54,
    "H + O2 -> OOH": 0.39,
    "OOH + H -> H2O2": 0.25,
    "OH + OH -> H2O + O": 0.01
}

# Barreras para procesos de ruptura molecular o desorción/deslizamiento débil
Ed = {
    "O2 -> 2O": 0.311,
    "OOH -> H + O2": 0.45,
    "OH -> H + O": 0.97
}

#voy a hacer una lista de todas las energias
Energias = list(Eb.values()) + list(Ec.values()) + list(Ed.values())
E_min = min(Energias)

#Aca voy a poner todo lo referente a las prob de mov/reaccion, esto depende de T, de las energias de
# activacion y reaccion. Y se define por la formula de Arrhenius

e = 1.602*10**-19 #carga del electron en C (se usa solo para sigma, no para el termino de Arrhenius: Ea ya esta en eV)
A = 10**13 #s^-1 cte prexponencial para el platino
V = 0 #todavia no se genero voltaje
def Arrhenius(Ea, V, T):
    k = 8.617*10**-5  # constante de Boltzmann en eV/K
    P = A * np.exp((-Ea)/(k*C*T))   # Ea y k ya estan en eV, asi que un electron cruzando V volts aporta "V" eV, sin multiplicar por e
    return P

def tau(T):
    """Duración de un paso MC en segundos, a esta T"""
    return 1 / Arrhenius(E_min, V, T)

#Guarda que esto no esta normalizado, eso puede dar problemas cuando vaya a sortear xq rando.random() me tira
#algo entre 0 y 1

def ArrNorm(Ea, V,T):
    k = 8.617*10**-5  # constante de Boltzmann en eV/K
    P = A * np.exp((-Ea)/(k*C*T))    # idem Arrhenius: sin el "e"
    return 1 - np.exp(-P*tau(T))   #normalizo con el tiempo de un paso MC, para que la probabilidad de que ocurra algo en un paso sea <1

transferencia = 0.5 #factor de transferencia, inventado, charlarlo con mati, se asume se cumple en las dos direcciones, entonces BV es una sola ec identica tanto para reacciones catodicas como anodicas
def BV(Ea, V, T):   #Butler-Volmer
    k = 8.617*10**-5  # constante de Boltzmann en eV/K
    P = A * np.exp((-Ea - transferencia*V)/(k*C*T))    # Ea y k ya estan en eV, asi que un electron cruzando V volts aporta "V" eV, sin multiplicar por e
    return P/Arrhenius(E_min, V, T)

def ProbBV(Ea, V, T):
    k = 8.617*10**-5  # constante de Boltzmann en eV/K
    P = A * np.exp((-Ea - transferencia*V)/(k*C*T))    # Ea y k ya estan en eV, asi que un electron cruzando V volts aporta "V" eV, sin multiplicar por e
    return 1 - np.exp(-P*tau(T))

#parametros iniciales
n_H_inicial = 1600
n_O2_inicial = 400

# Parámetros nuevos, junto a n_H_inicial / n_O2_inicial

flujo_H = 8    # cuántos H+ intentan entrar por paso
flujo_O2 = 2   # cuántos O2 intentan entrar por paso

# --- Probabilidades que NO dependen de V: ninguna de estas es un electron cruzando
# la interfaz metal/electrolito, son procesos puramente quimicos/difusivos entre
# especies neutras sobre el Pt. Se calculan UNA SOLA VEZ, antes del loop, con V=0,
# y se quedan asi para siempre (a proposito).
prob_mover_H = ArrNorm(Eb["H"], V, T)
prob_mover_O = ArrNorm(Eb["O"], V, T)
prob_mover_OH = ArrNorm(Eb["OH"], V, T)
prob_mover_O2 = ArrNorm(Eb["O2"], V, T)
prob_mover_H2O2 = ArrNorm(Eb["H2O2"], V, T)
prob_mover_OOH = ArrNorm(Eb["OOH"], V, T)

matriz = np.zeros(shape = (N, N))   #aca van a vivir las moleculas
membrana = np.zeros(shape = (N, N)) #aca van a vivir los protones

#Diccionario, atajo para referirse a cada tipo de particula por su nombre en vez de por un numero
particula = {
    "Pt" : 0,
    "Membrana" : 0,
    "H" : 1,
    "H2" : 2,
    "O" : 3,
    "H2O" : 4,
    "OH" : 5,
    "O2" : 6,
    "H2O2" : 7,
    "OOH" : 8
}

# Mapeo de número de partícula → símbolo para el archivo XYZ
# OVITO usa estos nombres para colorear y etiquetar
simbolo_xyz = {
    0: "Pt",   # Platino (vacío en la grilla)
    1: "H",    # Protones de Hidrógeno
    2: "N",    # H2  → usamos N como placeholder (OVITO lo va a mostrar como "N")
    3: "O",    # Oxígeno
    4: "O",    # H2O → mismo color que O pero más grande (ver abajo)
    5: "F",    # OH  → F como placeholder
    6: "C",    # O2  → C como placeholder
    7: "S",    # H2O2 → S como placeholder
    8: "Cl"    # OOH → Cl como placeholder
}

# Para que OVITO sepa el nombre real de cada tipo, usamos la propiedad "Particle Type"
# pero lo más simple es usar el campo de comentario del frame para documentarlo.
# El formato XYZ estándar que OVITO entiende es:
#
#   <N_atomos>
#   <comentario o propiedades>
#   <tipo> <x> <y> <z>
#   ...
#
# Usamos z=0 porque la simulación es 2D.

# reacciones laterales (O+O->O2, OH+OH->O): dos especies neutras combinandose en
# la superficie, sin ningun proton ni electron cruzando la interfaz -> NO dependen
# de V. Se calculan UNA SOLA VEZ, con V=0, y se quedan afuera del loop (a proposito).
reacciones = [
    (particula["O"], particula["O"], particula["O2"], ArrNorm(Ec["O + O -> O2"], V, T)),
    (particula["OH"], particula["OH"], particula["O"], ArrNorm(Ec["OH + OH -> H2O + O"], V, T))
]

# disociaciones: no se usa en ningun lado del loop principal (el bloque de
# "Disociar cosas" llama a ArrNorm(Ed[...], V, T) directo con el V del paso).
# Queda como estaba, es codigo muerto - no importa si esta adentro o afuera del loop.
disociaciones = [
    (particula["O2"], particula["O"], particula["O"], ArrNorm(Ed["O2 -> 2O"], V, T)),
    (particula["OOH"], particula["H"], particula["O2"], ProbBV(Ed["OOH -> H + O2"], V, T)),
    (particula["OH"], particula["H"], particula["O"], ProbBV(Ed["OH -> H + O"], V, T))
]


#Ubicamos particulas de H y O de forma aleatoria en la matriz, y las agregamos a la lista de moléculas activas.
protones = []                                          #los protones tienen su propia lista de protones
for _ in range(n_H_inicial):
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    if membrana[i, j] == particula["Membrana"]:  # solo colocar en celdas vacías
        membrana[i, j] = particula["H"]
        protones.append([particula["H"], i, j])

moleculas = []
for _ in range(n_O2_inicial):                          #tanto los O2 como el resto de moleculas van a la lista de moleculas
    i = random.randint(0, N - 1)
    j = random.randint(0, N - 1)
    if matriz[i, j] == particula["Pt"]:
        matriz[i, j] = particula["O2"]
        moleculas.append([particula["O2"], i, j])


# ============================================================
# EXPORTACIÓN A XYZ PARA OVITO
# ============================================================
# Nombres legibles que van a aparecer como comentario en cada frame
nombre_particula = {
    1: "H",
    2: "H2",
    3: "O",
    4: "H2O",
    5: "OH",
    6: "O2",
    7: "H2O2",
    8: "OOH"
}

xyz_file = open("simulacion.xyz", "w")

def escribir_frame_xyz(archivo, moleculas, protones, paso):
    """
    Escribe un frame en formato XYZ extendido que OVITO puede leer.
    Cada partícula se guarda con:
      - su nombre real (H, H2, O, etc.)   → columna 'Particle Type'
      - su posición x, y (z=0)
    El formato extendido permite pasarle el nombre como string, así OVITO
    puede colorear por especie correctamente.
    """
    n_atomos = len(moleculas) + len(protones)  # contar también los protones
    if n_atomos == 0:
        return

    # Cabecera del frame (formato XYZ extendido de OVITO)
    archivo.write(f"{n_atomos}\n")
    archivo.write(
        f'Lattice="{N} 0 0 0 {N} 0 0 0 5" '
        f'Properties=species:S:1:pos:R:3 '
        f'Time={paso}\n'
    )

    for mol in moleculas:
        tipo = mol[0]
        x    = float(mol[2])   # columna → eje X
        y    = float(mol[1])   # fila    → eje Y
        nombre = nombre_particula.get(tipo, "X")
        archivo.write(f"{nombre} {x:.2f} {y:.2f} 0.00\n")

    for prot in protones:
        x = float(prot[2])
        y = float(prot[1])
        archivo.write(f"H {x:.2f} {y:.2f} 6.00\n")   # z=1 → capa de membrana, por encima del Pt
# ============================================================


#lista de cosas que me interesa contar
aguas_por_paso = []
aguas_acumuladas_por_paso = []
peroxidos_por_paso = []
Volt = []
n_catodica_paso = []
n_anodica_paso = []


aguas = 0
cargas = 0
peroxidos = 0

for k in range(n):
    sigma = (cargas * e)/(N*N*d*d) #cargas por area, en C/m^2 (cargas es un conteo de electrones, hay que multiplicar por la carga elemental)
    V = sigma/(2*epsilon)*z
    Volt.append(V)    #Ahora que reaccionaron es un buen momento para considerar el aumento de carga positiva del Pt

    # V acaba de cambiar: recalculo aca, con el V fresco, todo lo que SI depende
    # de V (el mecanismo de carga cruzando la interfaz) para que lo usen el resto
    # de los bloques de este mismo paso (mover protones) y el proximo paso
    # (reacciones_H, arriba).
    reacciones_H = [
        (particula["H"], particula["O"], particula["OH"], ProbBV(Ec["H + O -> OH"], V, T)),
        (particula["OH"], particula["H"], particula["H2O"], ProbBV(Ec["OH + H -> H2O"], V, T)),
        (particula["H"], particula["O2"], particula["OOH"], ProbBV(Ec["H + O2 -> OOH"], V, T)),
        (particula["OOH"], particula["H"], particula["H2O2"], ProbBV(Ec["OOH + H -> H2O2"], V, T))
    ]

    # Borrar aguas del paso anterior
    aguas_este_paso = 0
    n_catodo = 0
    n_anodo = 0
    for i in range(N):
        for j in range(N):
            if matriz[i, j] == particula["H2O"]:
                matriz[i, j] = particula["Pt"]
                if [particula['H2O'], i, j] in moleculas:
                    moleculas.remove([particula['H2O'], i, j])

    #Antes de mover las particulas, voy a hacer reaccionar a los protones con lo que tienen justo abajo
    #(reacciones_H usa el V calculado al final del paso anterior - para k=0 es el V=0 inicial)
    for prot in protones[:]:
        i = prot[1]
        j = prot[2]
        t = matriz[i, j]
        if t != particula["Pt"]:
            for reaccion in reacciones_H:
                if (reaccion[0] == particula["H"] and reaccion[1] == t) or (reaccion[0] == t and reaccion[1] == particula["H"]):
                    if random.random() < reaccion[3]:
                        producto = reaccion[2]              # nombre claro, para la cosa que se genera

                        matriz[i, j] = producto             # meto el nuevo resultado en la matriz, limpio la membrana y su lista
                        protones.remove(prot)
                        membrana[i, j] = 0

                        for mol in moleculas:               # si la reaccion se dio, para la lista de moleculas que cayeron debajo del proton (en i,j) les reasigno el nombre en la lista
                            if mol[1] == i and mol[2] == j:
                                mol[0] = producto
                                break

                        cargas += 1; n_catodo += 1             # toda reaccion de este bloque es H+ + e-
                        if producto == particula['H2O']:    # cuento las reacciones que me interesen
                            aguas += 1; aguas_este_paso += 1
                        if producto == particula['H2O2']:
                            peroxidos += 1
                        break


    # Mover las partículas
    for mol in moleculas[:]:
        sorteo = random.random()

        s = mol[0]
        i = mol[1]
        j = mol[2]

        if s == particula["O"] and sorteo > prob_mover_O:
            continue
        if s == particula["OH"] and sorteo > prob_mover_OH:
            continue
        if s == particula["O2"] and sorteo > prob_mover_O2:
            continue
        if s == particula["H2O2"] and sorteo > prob_mover_H2O2:
            continue
        if s == particula["OOH"] and sorteo > prob_mover_OOH:
            continue

            #aca doy 20 intentos para mover la partícula, si no se mueve, se queda quieta. Esto es para evitar que queden atrapadas entre otras partículas.
            # Si la particula intenta atravesar la frontera, se aplica condición de frontera periódica (se "teletransporta" al otro lado).
        for _ in range(20):
            direccion = random.randint(0, 3)

            di = [-1, 1, 0, 0][direccion]
            dj = [0, 0, 1, -1][direccion]
            ni, nj = (i + di) % N, (j + dj) % N

            if matriz[ni, nj] != particula["Pt"]:
                continue

            matriz[i, j] = particula["Pt"]
            matriz[ni, nj] = s
            mol[1] = ni
            mol[2] = nj
            break

    #Mover protones
    for prot in protones[:]:
        sorteo = random.random()

        s = prot[0]
        i = prot[1]
        j = prot[2]

        if s == particula["H"] and sorteo > prob_mover_H:   #s solo puede ser H, asi que es medio que un if que se va a cumplir siempre, lo dejo por si llegase a agregar algo mas, pero no creo
            continue

        for _ in range(20):
            direccion = random.randint(0, 3)

            di = [-1, 1, 0, 0][direccion]
            dj = [0, 0, 1, -1][direccion]
            ni, nj = (i + di) % N, (j + dj) % N

            if membrana[ni, nj] != particula["Membrana"]:
                continue


            membrana[i, j] = particula["Membrana"]
            membrana[ni, nj] = s
            prot[1] = ni
            prot[2] = nj
            break


    # Combinaciones laterales
    for mol in moleculas[:]:
        s = mol[0]
        i = mol[1]
        j = mol[2]
        reacciono = False
        for (di, dj) in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
            if reacciono:
                break  # ← rompe el loop de vecinos si ya reaccionó
            ni, nj = (i + di) % N, (j + dj) % N
            if matriz[ni, nj] != particula["Pt"]:
                t = matriz[ni, nj]
                for reaccion in reacciones:
                    if (reaccion[0] == s and reaccion[1] == t) or (reaccion[0] == t and reaccion[1] == s):
                        if random.random() < reaccion[3]:
                            matriz[i, j] = reaccion[2]
                            matriz[ni, nj] = particula["Pt"]
                            mol[0] = reaccion[2]
                            if [t, ni, nj] in moleculas:
                                moleculas.remove([t, ni, nj])
                            if reaccion[2] == particula['O']:           #Ojo con esto, OH+OH=O+H2O pero por como esta escrito el codigo solo va a aparecer en la matriz el O, esto solo afectaria a la exportacion a ovito pero los resultados duros estan en las curvas y a eso no lo va a afectar
                                aguas += 1; aguas_este_paso += 1
                            reacciono = True
                            break  # ← rompe loop de reacciones

    # Disociar cosas — loop separado
    for mol in moleculas[:]:
        s = mol[0]
        i = mol[1]
        j = mol[2]
        if s == particula["O2"]:
            vecinos_libres = []
            for (di, dj) in [(-1, 0), (1, 0), (0, 1), (0, -1)]:
                ni, nj = (i + di) % N, (j + dj) % N
                if 0 <= ni < N and 0 <= nj < N:
                    if matriz[ni, nj] == particula["Pt"]:
                        vecinos_libres.append((ni, nj))
            if len(vecinos_libres) > 0:
                if random.random() < ArrNorm(Ed["O2 -> 2O"], V, T):
                    idx_vecino = random.randint(0, len(vecinos_libres) - 1)
                    ni, nj = vecinos_libres[idx_vecino]
                    matriz[i, j] = particula["O"]
                    matriz[ni, nj] = particula["O"]
                    mol[0] = particula["O"]
                    moleculas.append([particula["O"], ni, nj])

        if s == particula["OOH"]:
            if membrana[i, j] == particula["Membrana"]:
                if random.random() < ProbBV(Ed["OOH -> H + O2"], -V, T):    #negativo porque la barrera electrostatica esta favorecienda la disociacion
                    matriz[i, j] = particula["O2"]
                    membrana[i, j] = particula["H"]
                    mol[0] = particula["O2"]
                    protones.append([particula["H"], i, j])
                    cargas -= 1  ; n_anodo += 1                                         #El proton se fue hacia la membrana
                    

        if s == particula["OH"]:
            if membrana[i, j] == particula["Membrana"]:
                if random.random() < ProbBV(Ed["OH -> H + O"], -V, T):
                    matriz[i, j] = particula["O"]
                    membrana[i, j] = particula["H"]
                    mol[0] = particula["O"]
                    protones.append([particula["H"], i, j])
                    cargas -= 1 ; n_anodo += 1                                             #El proton se fue hacia la membrana


    #Meter flujo de particulas nuevas
    #Protones
    for _ in range(flujo_H):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if membrana[i, j] == particula["Membrana"]:
                membrana[i, j] = particula["H"]
                protones.append([particula["H"], i, j])

    #Oxigeno
    for _ in range(flujo_O2):
            i = random.randint(0, N - 1)
            j = random.randint(0, N - 1)
            if matriz[i, j] == particula["Pt"]:
                matriz[i, j] = particula["O2"]
                moleculas.append([particula["O2"], i, j])

    # ── Escribir el frame de este paso al archivo XYZ ──
    escribir_frame_xyz(xyz_file, moleculas, protones, k)

    #Registrar cantidad de aguas formadas en este paso
    aguas_por_paso.append(aguas_este_paso)
    aguas_acumuladas_por_paso.append(aguas)
    peroxidos_por_paso.append(peroxidos)
    n_catodica_paso.append(n_catodo)
    n_anodica_paso.append(n_anodo)
    # Progreso en consola
    if k % 50 == 0:
        print(f"Paso {k}/{n}  |  Moléculas activas: {len(moleculas)}  |  H2O formadas: {aguas}")

xyz_file.close()
print(f"\n✓ Simulación terminada. Aguas formadas: {aguas}, cargas liberadas: {cargas}")
print("✓ Archivo 'simulacion.xyz' generado — abrilo con OVITO.")

# Eje de tiempo real
pasos = np.arange(n)
tiempo_real = pasos * tau(T)

# Gráfico 1: H2O y H2O2 acumuladas en la misma figura
plt.figure(figsize=(10, 6))
plt.plot(tiempo_real, aguas_acumuladas_por_paso, label="H₂O acumuladas", linewidth=2)
plt.plot(tiempo_real, peroxidos_por_paso, label="H₂O₂ acumuladas", linewidth=2)
plt.xlabel("Tiempo simulado (s)")
plt.ylabel("Cantidad acumulada")
plt.title("Evolución de H₂O y H₂O₂ acumuladas")
plt.legend()
plt.grid()
plt.show()

# Gráfico 2: fracción H2O2 / H2O
aguas_arr = np.array(aguas_acumuladas_por_paso, dtype=float)
peroxidos_arr = np.array(peroxidos_por_paso, dtype=float)
 
# evitar división por cero
fraccion = np.divide(peroxidos_arr, aguas_arr,
                      out=np.zeros_like(peroxidos_arr),
                      where=aguas_arr != 0)
 
plt.figure(figsize=(10, 6))
plt.plot(tiempo_real, fraccion, color="gold", linewidth=2)
plt.xlabel("Tiempo simulado (s)")
plt.ylabel("Fracción H₂O₂ / H₂O")
plt.title("Selectividad: relación H₂O₂ / H₂O acumulada")
plt.grid()
plt.show()

#Grafico 3: Voltaje generado
plt.figure(figsize=(10, 6))
plt.plot(tiempo_real, Volt, color="green", linewidth=2)
plt.xlabel("Tiempo simulado (s)")
plt.ylabel("Voltaje (V)")
plt.title("Voltaje generado por la reacción")
plt.grid()
plt.show()

# Gráfico 4: corrientes faradaicas
W = 50                                   # ventana de promedio, en pasos
area = N*N*d*d                           # área del electrodo, m²
ker = np.ones(W)/W
J_cat = e*np.convolve(n_catodica_paso, ker, 'valid')/(tau(T)*area)   # A/m²
J_an  = e*np.convolve(n_anodica_paso,  ker, 'valid')/(tau(T)*area)
J_net = J_cat - J_an
t_J = tiempo_real[W-1:]

plt.figure(figsize=(10, 6))
plt.plot(t_J, J_cat, label="catódica", linewidth=2)
plt.plot(t_J, J_an, label="anódica", linewidth=2)
plt.plot(t_J, J_net, label="neta", linewidth=2, color="black")
plt.xlabel("Tiempo simulado (s)")
plt.ylabel("Densidad de corriente (A/m²)")
plt.title("Corrientes faradaicas")
plt.legend()
plt.grid()
plt.show()