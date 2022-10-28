# Parâmetros
 - Épocas: 140
 - Batch: 16 (625 amostras por batch/época)
 - Imagens: 201 (2048x2048)
 - Split Train/Test: A separação foi randômica
    - 80% treinamento, 20% teste
 - Tempo para treinamento: 42h
 - Tempo para teste: 4h

# Resultados

 - Precisão global: 88,16%

# Sugestões dos professores

 - Verificar outra loss que lide melhor com desbalancemanto de classes;
 - Verificar o tamanho do Batch. 625 amostras por época parece um número muito grande;
 - Ao fazer o split (80/20), separar melhor as imagens para cobrir igualmente no treinamento e no teste;
 - Verificar a possibilidade de fazer mais data augmentation nas classes que tem menos amostras (piscina, solo, etc);
 - Talvez, remover data augmentation

1. Verificar outra loss que lide melhor com desbalancemanto de classes;
   [] CrossEntropyLoss: utilizar pesos para cada classe
      - INS (Inversal Number of Samples)
      - etc
      - Weighted Cross Entropy Loss: 
        https://towardsdatascience.com/deep-learning-with-weighted-cross-entropy-loss-on-imbalanced-tabular-data-using-fastai-fe1c009e184c
   [] FocalLoss + DiceLoss

2. Data Augmentation
   [] Utilizar a biblioteca albumentation
   [] Fazer Data Augmentation como pré-processamento

3. Descontinuidades (pós-processamento)
   [] CRF 
   [] https://www.youtube.com/watch?v=HrGn4uFrMOM

4. Ao fazer o split (80/20), separar melhor as imagens para cobrir igualmente no treinamento e no teste;
   [] KFold

