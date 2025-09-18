import torch
import torch.nn as nn
import torch.nn.functional as F

class ScoreRestraintAttentionPooling(nn.Module):
    """
    Implementación del mecanismo Score-restraint Attention Pooling.
    
    Este módulo calcula una representación de tamaño fijo para una secuencia
    ponderando los estados ocultos de la secuencia según las puntuaciones de
    calidad predichas en niveles inferiores.
    """
    def __init__(self, num_scores: int, hidden_dim: int):
        """
        Inicializador del módulo.
        
        Args:
            num_scores (int): El número total de puntuaciones de bajo nivel
                              concatenadas (e.g., 1 de fonema + 3 de palabra = 4).
            hidden_dim (int): La dimensionalidad de los estados ocultos de entrada.
                              (No es estrictamente necesario para el __init__ pero es
                               buena práctica para la claridad).
        """
        super().__init__()
        # Capa lineal que implementa W y b de la Fórmula 12
        # Proyecta el vector de puntuaciones a un único logit de saliencia.
        self.score_to_salience = nn.Linear(num_scores, 1)

    # (Continuación de la clase ScoreRestraintAttentionPooling)

    def forward(self, 
                hidden_states: torch.Tensor, 
                phone_scores: torch.Tensor,
                mdd_scores: torch.Tensor, 
                word_scores: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Define el forward pass del módulo.
        
        Args:
            hidden_states (torch.Tensor): Tensor de estados ocultos de la secuencia.
                                          Shape:
            phone_scores (torch.Tensor): Tensor de puntuaciones de precisión de fonemas.
                                         Shape:
            word_scores (torch.Tensor): Tensor de puntuaciones de palabra (precisión, acento, total).
                                        Shape:
            mask (torch.Tensor, optional): Máscara de atención para manejar el padding.
                                           Shape:. Los valores True indican
                                           posiciones válidas. Defaults to None.
        
        Returns:
            torch.Tensor: El vector agregado para cada elemento del lote.
                          Shape:
        """
        
        # --- Paso 1: Preparación y Concatenación de Puntuaciones ---
        # Corresponde a la entrada de la Fórmula 12: [p_i, w_i^0, w_i^1, w_i^2]
        # Concatenamos los tensores de puntuaciones a lo largo de la última dimensión.
        # Shape: + ->

        if word_scores is None:
            # Si no se proporcionan puntuaciones de palabra, usamos solo las de fonema y MDD.
            combined_scores = torch.cat([phone_scores, mdd_scores], dim=-1)
        else:
            # Concatenamos todas las puntuaciones disponibles.
            combined_scores = torch.cat([phone_scores, mdd_scores, word_scores], dim=-1)
        
        # --- Paso 2: Cálculo de la Saliencia (Fórmula 12) ---
        # s_i = GELU(W(...) + b)
        # Aplicamos la capa lineal y la activación GELU.
        # Shape: ->
        salience_scores = self.score_to_salience(combined_scores)
        salience_scores = F.gelu(salience_scores)
        
        # --- Paso 3: Cálculo de los Pesos de Atención (Fórmula 13) ---
        # α_i = softmax(s_i)
        # Antes de aplicar softmax, es crucial manejar el padding para que los
        # tokens de padding no reciban atención.
        if mask is not None:
            # Añadimos una dimensión para el broadcasting: ->
            mask = mask.unsqueeze(-1)
            # Donde la máscara es False (padding), asignamos un valor muy negativo
            # para que su probabilidad en el softmax sea cercana a cero.
            salience_scores = salience_scores.masked_fill(mask == 0, -1e9)
                

        # Aplicamos softmax a lo largo de la dimensión de la secuencia (T).
        # El squeeze(-1) elimina la última dimensión de tamaño 1.
        # Shape: ->
        attention_weights = F.softmax(salience_scores, dim=0).squeeze(-1)
        
        # --- Paso 4: Agregación Ponderada (Fórmula 14) ---
        # h_agg = (1/N) * Σ(α_l * h_utt^l)
        
        # Para la multiplicación ponderada, necesitamos que los pesos tengan la forma
        # para que se puedan multiplicar con los hidden_states.
        # Shape: ->
        attention_weights_expanded = attention_weights.unsqueeze(-1)
        
        # Multiplicación elemento a elemento (ponderación).
        # Shape: * ->
        weighted_hidden_states = hidden_states * attention_weights_expanded
        
        # Suma a lo largo de la dimensión de la secuencia para agregar.
        # Shape: ->
        aggregated_vector = torch.sum(weighted_hidden_states, dim=0)
        
        
        # Aplicación de la normalización por longitud (1/N).
        # N es el número de elementos no acolchados en cada secuencia del lote.
        if mask is not None:
            # Sumamos la máscara para obtener la longitud real de cada secuencia.
            # Shape: ->
            sequence_lengths = mask.sum(dim=0)
            # Añadimos una dimensión para la división: ->
            sequence_lengths = sequence_lengths.unsqueeze(-1)
            # Evitar división por cero si hay secuencias vacías.
            sequence_lengths = torch.clamp(sequence_lengths, min=1)
        else:
            # Si no hay máscara, todas las secuencias tienen la misma longitud.
            sequence_lengths = hidden_states.shape[0]

        # Normalización final.
        # Shape: / ->
        final_aggregated_vector = aggregated_vector / sequence_lengths
                
        return final_aggregated_vector