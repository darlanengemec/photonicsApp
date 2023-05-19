import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from model_training import train_regression_model

# Treinar o modelo e obter o modelo treinado
model = train_regression_model()

# # Carregar o modelo pré-treinado
# model_path = "var/model.pkl.xz"
# model = joblib.load(model_path)

# Função para realizar a previsão
def fazer_previsao(h, r, lambdai, lambdaf, p):
    n_samples = int((lambdaf - lambdai) / p) + 1

    x1 = []
    for i in range(n_samples):
        x1.append([h])

    x2 = []
    for i in range(n_samples):
        x2.append([r])

    x3 = []
    for i in range(n_samples):
        valor = int(lambdai + i * p)
        x3.append([valor])

    x = np.concatenate((x1, x2, x3), axis=1)
    x = pd.DataFrame(x, columns=['altura', 'raio', 'lambda'])

    # Fazer a previsão com o modelo carregado
    prediction = model.predict(x)

    # Plotar o gráfico
    fig, ax = plt.subplots()
    ax.plot(x['lambda'], prediction, color='blue')
    ax.set_title('Extra Tree Regressor Predict')
    ax.set_xlabel('Wavelength [nm]')
    ax.set_ylabel('Scattering Cross Section x10^-14[a.u]')
    # Adicionar anotações ao gráfico
    max_scattering = prediction.max()
    max_wavelength = x['lambda'][prediction.argmax()]

    ax.annotate(f"Altura: {h} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering), color='red')
    ax.annotate(f"Raio: {r} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering * 0.9), color='red')
    ax.annotate(f"Comprimento de onda do pico: {max_wavelength} nm", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength, max_scattering), color='red')
    ax.annotate(f"Valor do pico: {max_scattering:.4f} [a.u]", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength, max_scattering * 0.95), color='red')

    return fig

# Configurações do Streamlit
st.set_page_config(page_title="Predict Scattering API", layout="wide")

# Título da página e logo
st.title("Predict Scattering API")
logo_image = "/Users/MacBarroso/PycharmProjects/photonicsApp/assets/images/logo_fotonica.jpeg"
st.image(logo_image)  #, use_column_width=True)

# Dividir a tela em duas colunas
col1, col2 = st.columns(2)

# Entradas do usuário
h = col1.number_input("Enter the height of the cylindrical gold nanostructure in nm:", value=0.0)
r = col1.number_input("Enter the radius of the cylindrical gold nanostructure in nm:", value=0.0)
lambdai = col1.number_input("Enter the initial applied wavelength in nm:", value=0.0)
lambdaf = col1.number_input("Enter the final applied wavelength in nm:", value=0.0)
p = col1.number_input("Enter the step of the applied wavelength in nm:", value=0)

# Verificar se todos os campos foram preenchidos
if h != 0.0 and r != 0.0 and lambdai != 0.0 and lambdaf != 0.0 and p != 0:
    # Fazer a previsão
    fig = fazer_previsao(h, r, lambdai, lambdaf, p)

    # Exibir o valor do pico de scattering
    valor_pico = fig.gca().get_lines()[0].get_ydata().max()
    col2.pyplot(fig)

    # Exibir o valor do pico de scattering
    col2.write(f"The scattering peak value for this nanostructure is: {valor_pico:.4f} [a.u]")
