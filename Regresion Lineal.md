Nuestra funcion va a ser una linea que se aproxime lo mas posible a todos los datos de entrenamiento. 

Dada una lista de features $x_{1}\dots x_{N}$ podemos ver una aproximacion de los datos de salida $z_{1}\dots z_{M}$. 

Esta aproximacion se hace con la matriz de pesos $W$ donde cada dato de salida $z_{i}$ es una combinacion lineal del vector de features por el vector de pesos. 
$z_{i} = x_{1}*w_{1} + \dots + x_{N}*w_{N}$. 

## Objetos Matemáticos
- $X \in \mathbb{R}^{P \times N+1}$: Matriz de features. Para cada dato de entrenamiento tenemos una lista de features $x_{1}\dots x_{N}$ y un 1 al final.
- $W \in \mathbb{R}^{N+1 \times M}$: Matriz de pesos. Las filas de entrenamiento de $X$ comparten la matriz de pesos. Para los features $x_{1}\dots x_{N}$ de la fila 1 de $X$ existen las $M$ columnas de $W$.

$X = \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1N} & 1\\    x_{21} & x_{22} & \dots & x_{2N} & 1\\ \vdots & \vdots &  & \vdots & \vdots\\ x_{P1} & x_{P2} & \dots & x_{PN} & 1\end{bmatrix}$

$W = \begin{bmatrix} w_{11} & w_{12} & \dots & w_{1M}\\    w_{21} & w_{22} & \dots & w_{2M}\\ \vdots & \vdots &  & \vdots\\ w_{(N+1)1} & w_{(N+1)2} & \dots & w_{(N+1)M}\end{bmatrix}$

$X.W =  \begin{bmatrix} x_{11} & x_{12} & \dots & x_{1N} & 1\\    x_{21} & x_{22} & \dots & x_{2N} & 1\\ \vdots & \vdots &  & \vdots & \vdots\\ x_{P1} & x_{P2} & \dots & x_{PN} & 1\end{bmatrix} \begin{bmatrix} w_{11} & w_{12} & \dots & w_{1M}\\    w_{21} & w_{22} & \dots & w_{2M}\\ \vdots & \vdots &  & \vdots\\ w_{(N+1)1} & w_{(N+1)2} & \dots & w_{(N+1)M}\end{bmatrix}$

- Cada dato de entrenamiento es una fila de X.
- Cada feature es un dato particular del dato de entrenamiento (metros cuadrados, banos, etc).
- Cada $w_{ij}$ es un peso inicializado al azar.
- Las matriz resultante los $M$ datos de salida para los respectivos datos de entrada.

## Gradiente
Usamos el gradiente de la funcion de cuadrados minimos para actualizar los pesos. 
$$\triangle w =  X^t \cdot D$$
donde:
- $D = Z - Y = Z - X \cdot W$
- $Y = X \cdot W$
- $Z$ es el conjunto real datos de salida.

$Z = \begin{bmatrix} z_{11} & z_{12} & \dots & z_{1M}\\    z_{21} & z_{22} & \dots & z_{2M}\\ \vdots & \vdots &  & \vdots\\ z_{P1} & z_{P2} & \dots & z_{PM}\end{bmatrix}$

