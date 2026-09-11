import random


def load_dataset(st):
    return st.file_uploader("Escolha sua base de dados", type='jsonl')

def load_model():
    """
    Carregamento do Modelo
    """
    return None



def train_new_model( epochs: int = 10, learning_rate: float = 0.001, batch_size: int = 4, 
                    penalty_input: float = 0.1, temperature_input: float = 0.7):
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
        'temperature': temperature_input,
        'penality': penalty_input,
        'loss_history': loss_history
    }

    return model, loss_history



def continue_training_model(model, add_epochs: int = 5, 
                            learning_rate: float = 0.001, batch_size: int = 4, 
                            temperature_input: float = 0.7, penalty_input: float = 0.1):
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
    model['temperature'] = temperature_input
    model['penalty'] = penalty_input

    # Adicionamento de perdas durante o re-treinamento
    model['loss_history'].extend(aditional_loss)

    return model, aditional_loss