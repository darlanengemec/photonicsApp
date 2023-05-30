import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Carregar o modelo pré-treinado
model_path = "var/model.pkl.xz"
model = joblib.load(model_path)

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

    ax.annotate(f"Height: {h} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering), color='red', fontsize=7)
    ax.annotate(f"Radius: {r} nm", xy=(lambdai, max_scattering), xytext=(lambdai, max_scattering * 0.9), color='red', fontsize=7)
    ax.annotate(f"Wavelength peak: {max_wavelength} nm", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength + 250, max_scattering), color='red', fontsize=7)
    ax.annotate(f"Scattering peak: {max_scattering:.4f} [a.u]", xy=(max_wavelength, max_scattering),
                xytext=(max_wavelength + 250, max_scattering * 0.9), color='red', fontsize=7)

    return fig

# Configurações do Streamlit
st.set_page_config(page_title="Predict Scattering API", layout="wide")

# Título da página e logo
st.title("Predict Scattering API")

# Redimensionar a imagem
logo_image = Image.open("assets/images/logo_fotonica.jpeg")
logo_image = logo_image.resize((200, 200))  # Ajuste as dimensões conforme necessário

# Dividir a tela em duas colunas
col1, col2 = st.columns(2)

col1.image(logo_image)

# Entradas do usuário
h = col1.text_input("Enter the height of the cylindrical gold nanostructure in nm:", key="h")
if h and h.isnumeric():
    h = float(h)
r = col1.text_input("Enter the radius of the cylindrical gold nanostructure in nm:", key="r")
if r and r.isnumeric():
    r = float(r)
lambdai = col1.text_input("Enter the initial applied wavelength in nm:", key="lambdai")
if lambdai and lambdai.isnumeric():
    lambdai = float(lambdai)
lambdaf = col1.text_input("Enter the final applied wavelength in nm:", key="lambdaf")
if lambdaf and lambdaf.isnumeric():
    lambdaf = float(lambdaf)
p = col1.text_input("Enter the step of the applied wavelength in nm:", key="p")
if p and p.isnumeric():
    p = int(p)

# Botão de reset
reset_button = col1.button("Reset")

if reset_button:
    # Redefinir todas as entradas
    h = None
    r = None
    lambdai = None
    lambdaf = None
    p = None

# Verificar se todas as caixas de entrada estão preenchidas e são números válidos
if h is not None and r is not None and lambdai is not None and lambdai != '' and lambdaf is not None and lambdaf != '' and p is not None and p != '':
    # Converta lambdai, lambdaf e p para os tipos corretos
    lambdai = float(lambdai)
    lambdaf = float(lambdaf)
    p = int(p)

    # Fazer a previsão
    fig = fazer_previsao(h, r, lambdai, lambdaf, p)
    # Exibir o valor do pico de scattering
    valor_pico = fig.gca().get_lines()[0].get_ydata().max()

    # Subir o gráfico e a mensagem
    col2.empty()

    # Ajustar a altura do gráfico para 300 pixels
    fig.set_size_inches(5, 4)  # Ajuste as dimensões conforme necessário

    # Exibir o gráfico
    col2.pyplot(fig)

    # Centralizar a mensagem abaixo do gráfico
    col2.text("")  # Adicionar espaço em branco
    col2.markdown(
        f'<p style="text-align: center;">The scattering peak value for this nanostructure is:</p>',
        unsafe_allow_html=True
    )
    col2.markdown(
        f'<p style="text-align: center;">{valor_pico:.4f} [a.u]</p>',
        unsafe_allow_html=True
    )


# # Entradas do usuário
# h = col1.text_input("Enter the height of the cylindrical gold nanostructure in nm:", key="h")
# if h and h.isnumeric():
#     h = float(h)
#     r = col1.text_input("Enter the radius of the cylindrical gold nanostructure in nm:", key="r")
#     if r and r.isnumeric():
#         r = float(r)
#         lambdai = col1.text_input("Enter the initial applied wavelength in nm:", key="lambdai")
#         if lambdai and lambdai.isnumeric():
#             lambdai = float(lambdai)
#             lambdaf = col1.text_input("Enter the final applied wavelength in nm:", key="lambdaf")
#             if lambdaf and lambdaf.isnumeric():
#                 lambdaf = float(lambdaf)
#                 p = col1.text_input("Enter the step of the applied wavelength in nm:", key="p")
#                 if p and p.isnumeric():
#                     p = int(p)
#                     # Verificar se todas as caixas de entrada estão vazias
#                     if h is not None and r is not None and lambdai is not None and lambdaf is not None and p is not None:
#                         # Fazer a previsão
#                         fig = fazer_previsao(h, r, lambdai, lambdaf, p)
#
#                         # Exibir o valor do pico de scattering
#                         valor_pico = fig.gca().get_lines()[0].get_ydata().max()
#
#                         # Subir o gráfico e a mensagem
#                         col2.empty()
#
#                         # Ajustar a altura do gráfico para 300 pixels
#                         fig.set_size_inches(5, 4)  # Ajuste as dimensões conforme necessário
#
#                         # Exibir o gráfico
#                         col2.pyplot(fig)
#
#                         # Centralizar a mensagem abaixo do gráfico
#                         col2.text("")  # Adicionar espaço em branco
#                         col2.markdown(
#                             f'<p style="text-align: center;">The scattering peak value for this nanostructure is:</p>',
#                             unsafe_allow_html=True
#                         )
#                         col2.markdown(
#                             f'<p style="text-align: center;">{valor_pico:.4f} [a.u]</p>',
#                             unsafe_allow_html=True
#                         )
# #
#
#
#
#
#
#
#
#
