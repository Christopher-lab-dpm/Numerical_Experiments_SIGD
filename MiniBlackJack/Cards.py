import torch


def draw_cards(num_draws: int,
               num_sims: int,
               max_card: int = 5) -> torch.Tensor:
    
    """
    This function returns a tensor with shape (num_sims, num_draws) 
    where each row represents a simulation of drawing from a deck of cards with replacement.

    The deck consists of ace to max_card (inclusive), we neglect the suit as they do not matter for the game.
    And the probability of drawing one suit or another is the same for each card.


    Args:

        num_draws: The number of simulations we wish to do
        num_sims: The number of draws we wish to do for each simulation 
        max_cards: The highest value of the cards we wish to be in play in the game.

    Output:
        Returns a tensor with shape (num_sims, num_draws) 

    """
    return torch.randint(low=1, high = max_card + 1, size = (num_sims, num_draws), dtype=torch.int32)



def partial_sums(drawn_cards : torch.Tensor ) -> torch.Tensor:
    """
    Given a matrix which represents the number of simulations by the number of cards drawn in each simulations,
    give a partial sum of the cards values for each simualtion.

    Ex: Sim1 = [1,2,3,5] --> output = [1, 3, 6, 11] 

    Args:
        drawn_cards: Tensor of shape (num_sims, num_draws), as output by draw_cards

    Return:
        Tensor of shape (num_sims, num_draws) where the i'th row and j'th column corresponds to
        the j'th partial sum of the i'th simulation 
    """
    triangular_shape =  (drawn_cards.shape[1],  drawn_cards.shape[1]) # num_draws x num_draws 
    upper_triangular = torch.triu(torch.ones(triangular_shape))
    partial_sums = torch.matmul(drawn_cards.float() , upper_triangular.float())
    
    return partial_sums.int()
