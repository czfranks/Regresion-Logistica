import pandas as pd
import sklearn
import matplotlib.pyplot as plt

from sklearn.decomposition import KernelPCA

from sklearn.linear_model import LogisticRegression

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

if __name__ == "__main__":
    dt_heart = pd.read_csv('heart.csv')

    print(dt_heart)

    #COMPARAREMOS UNA REGRESION LOGISTICA SIMPLE Y R. LOGISTICA CON KERNEL PCA APLICADO

    #TARGET es la caracteristica que nos indica si el paciente, va a tener
    #        o no una complicacion cardiaca {0 , 1}

    #selecciono todas la caracteristicas de heart.csv, menos el target
    caracteristicas  = dt_heart.drop(['target'], axis=1)
    target = dt_heart['target']

    #aplican un scalamiento a los ejemplos de entrenamiento
    # z = ____X_ - _mean(X)___
    #       max(X) - min(X)
    caracteristicas = StandardScaler().fit_transform(caracteristicas)

    #escojo mi conjunto de entrenamiento
    #test_size = 0.3 -> el 30% usaremos para datos de testeo
    #random_state = 42, especificamos un numero para en cada ejecucion tengamos los mismos datos
    #                   escogidos aleatoriamente
    X_train, X_test, y_train, y_test = train_test_split(caracteristicas, target, test_size=0.3, random_state=42)
    X_train_copy = X_train

    #por que uso un kernelPCA?
    #               *este tipo de kernel nos ayuda a reducir la dimensionalidad( agrupando o no, las caracteristicas
    #                que no influyen mucho en el modelo )
    #es recomendable usar esta tecnica cuando tenemos muchas caracteristicas que pasan desapercibidas en los calculos
    kpca = KernelPCA(n_components=4, kernel='linear' )
    #X_train tendra una agrupacion(reduccion de dimensionalidad), por lo tanto
    #tendremos un mejor ejemplo de entrenamiento
    kpca.fit(X_train)

    #Usamos una regresion logistica sin kernelPCA,
    logisticSinKernel = LogisticRegression()
    logisticSinKernel.fit(X_train, y_train)

    dt_train = kpca.transform(X_train)
    dt_test = kpca.transform(X_test)

    logistic = LogisticRegression(solver='lbfgs')
    logistic.fit(dt_train, y_train)

    print("SCORE KPCA(kernel lineal): ", logistic.score(dt_test, y_test))
    print("SCORE Logistica lbfgs: ", logisticSinKernel.score(X_test, y_test))

    """
        SCORE KPCA(kernel lineal):  0.8214285714285714       #tenemos mejor prediccion con un kernel lineal 
        SCORE Logistica lbfgs:  0.8051948051948052
        
        SCORE Kernel Polynomial:  0.7987012987012987         #kernel polinomial no se ajusta mejor q un kernel lineal
                                                             #segun los ejemlos de entrenamiento 
    """

    #CONCLUSION: En este ejemplo de entrenamiento, aplicar kernel a los datos de entrada
    #            mejoro la presicion de nuestro modelo de R. Logistica, pero no siempre
    #            es de esta manera, y depende mucho del tipo de kernel a usar, en este caso
    #            un kernel lineal, nos dio mejore resultados

    #verificamos q para estos ejemlos de entrenamiento un kernel polinomial no nos da mejores resultados
    kpca2 = KernelPCA(n_components=4, kernel='poly')
    kpca2.fit(X_train_copy)
    dt_train2 = kpca2.transform(X_train_copy)
    dt_test2 = kpca2.transform(X_test)
    kernel_poly = LogisticRegression(solver='lbfgs')
    kernel_poly.fit(dt_train2, y_train)

    print("SCORE Kernel Polynomial: ", kernel_poly.score(dt_test2, y_test))
    """
    SCORE Kernel Polynomial:  0.7987012987012987
    """