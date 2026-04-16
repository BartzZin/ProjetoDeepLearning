# Projeto Perceptron com Scikit-learn

Projeto montado a partir do enunciado `UTFPR_pos_atividade_2_perceptron.pdf`, usando a implementacao oficial `sklearn.linear_model.Perceptron`.

## Base tecnica

- Modelo utilizado: `sklearn.linear_model.Perceptron`
- Treinamento por epocas com `partial_fit`
- Classes binarias `-1` e `1`
- Parametros principais alinhados com a documentacao oficial:
  - `eta0`
  - `max_iter`
  - `fit_intercept`
  - `shuffle`
  - `random_state`

## Estrutura

- `main.py`: executa todas as bases em `Bases`
- `src/iris_datasets.py`: gera as versoes binarias do dataset Iris
- `src/perceptron.py`: leitura dos CSVs e treinamento com `sklearn`
- `src/reporting.py`: geracao dos arquivos de saida
- `Resultados/`: artefatos produzidos na execucao

## Como executar

```powershell
py -3 main.py
```

Ou abra a interface grafica:

```powershell
py -3 gui.py
```

## Dependencias

```powershell
py -3 -m pip install -r requirements.txt
```

## Arquivos gerados

- `*_predicoes.csv`: classe real e classe prevista por amostra
- `*_historico_epocas.csv`: erros, acuracia, `coef_` e `intercept_` por epoca
- `*_resumo.json`: resumo final do treinamento

## Iris binario

O projeto tambem gera automaticamente duas bases derivadas do `Iris` em `Bases`:

- `iris_binario_4d.csv`: usa os quatro atributos originais
- `iris_binario_2d.csv`: usa `PetalLength` e `PetalWidth`

Nessas bases, `D = 1` representa `setosa` e `D = -1` representa `versicolor + virginica`.

## Observacao

As bases fornecidas ja possuem a coluna `Bias`, entao o projeto desabilita `fit_intercept` nesses casos para evitar duplicidade de vies. A base `xor.csv` continua sendo o caso esperado de nao convergencia para um Perceptron linear.

## Duvidas me chamar no Discord 🔥🔥 - Stuxnet#8096
