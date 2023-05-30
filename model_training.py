import os
import lzma
import pandas as pd
from pycaret.regression import setup, compare_models, create_model, tune_model, finalize_model
import joblib


def train_regression_model():
    # Importando Dataset
    data_regression = pd.read_csv('/Users/MacBarroso/PycharmProjects/pythonProject3/dados_1.CSV', sep=';')

    # Separando os dados de treino e teste
    data_train = data_regression.query('altura!=300.0 or raio!=50.0 and raio!=150.0')
    data_test_50 = data_regression.query('altura==300.0 and raio==50.0')
    data_test_150 = data_regression.query('altura==300.0 and raio==150.0')

    # Definindo os parâmetros de treinamento
    reg = setup(data_train, train_size=0.8, session_id=0)

    # Treinando e testando os modelos e avaliando os melhores quanto as métricas
    best = compare_models()

    # Criando o modelo com melhores métricas
    rf = create_model('rf')

    # Ajustando os hiperparâmetros do modelo
    tuned_rf = tune_model(rf, optimize='mse', n_iter=3, fold=3)

    best_model = rf if 'tuned' not in globals() or tuned_rf.best_score_ is None or tuned_rf.best_score_ >= rf.score(
        data_train.drop('medida', axis=1), data_train['medida']) else tuned_rf

    # Finalizando e salvando o melhor modelo
    final_model = finalize_model(best_model)

    # Salvando o modelo usando compressão LZMA
    joblib.dump(final_model, 'var/model.pkl.xz', compress=('xz', 6))

    return final_model
