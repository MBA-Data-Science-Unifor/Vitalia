import random


def load_dataset(st):
    return st.file_uploader("Escolha sua base de dados", type='jsonl')

def load_model():
    """
    Carregamento do Modelo
    """
    return None



def train_new_model(dataset, epochs, learning_rate, batch_size, temperature, penalty):
    """
    Simulação do treinamento
    """    
    loss_history = []
    loss = 1.0

    for epoch in range(int(epochs)):
        # queda por qualquer ruído
        loss = loss * 0.7 + random.uniform(-0.05, 0.05)
        loss = max(0.01, loss)
        loss_history.append(loss)

    # retorno do modelo
    model = {
        'trained': True,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'temperature': temperature,
        'penality': penalty,
        'loss_history': loss_history
    }

    return model, loss_history



def continue_training_model(model, dataset, add_epochs, learning_rate, batch_size, temperature, penalty):
    """
    Simula o Re-Treinamento
    """

    # Retorno do ultimo historico de perda conforme o modelo
    last_loss = model['loss_history'][-1] if ('loss_history' in model and model['loss_history']) else 1.0

    aditional_loss = []
    loss = last_loss

    for epoch in range(int(add_epochs)):
        # queda por qualquer ruído
        loss = loss * 0.8 + random.uniform(-0.03, 0.03)
        loss = max(0.01, loss)
        aditional_loss.append(loss)

    model['epochs'] += add_epochs
    model['learning_rate'] = learning_rate
    model['batch_size'] = batch_size
    model['temperature'] = temperature
    model['penalty'] = penalty

    # Adicionamento de perdas durante o re-treinamento
    model['loss_history'].extend(aditional_loss)

    return model, aditional_loss