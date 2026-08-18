import torch
from typing import Any



def Player_Policy(init_state_space: tuple[int , int],
                    partial_sums: torch.Tensor,
                    policy_probabilities: torch.Tensor,
                    max_hand_value: int = 11) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """ 
    Given the initial state state space, and the simulation of drawn cards
    from a deck (perhaps infitnite oR finite), the player policy probabilities
    are used to determine (either deterministically or stochastically) whether to
    hit or stay. Once the player stays or busts the remaining values are ignored.
   
    Args:
        init_state_space: Tuple indicating the current hand of the player  
                            and the dealer's hand as observed by the player. This
                            is the initial setup of the game.

        partial_sums: Tensor of size (num_sims , num_draws) which is the partial
                        sums over the simulations of the future cards to be drawn\

        policy_probabilities: Tensor of size 
        ( max_player_hand  - min_player_hand + 1 ,   max_observed_dealer_hand  - min_observed_dealer_hand + 1)
        which encodes the entire discretized state space of the mini-blackjack game and each entry
        (i,j) of the tensor corresponds to the probability of hitting.

            Example:   (cur_sum, dealer_sum) =  (5 , 6)      
                        policy_probabilities[5,6] = 0.1 (hit with prob 0.1)

        max_hand_value: Maximum valid hand value a player can have before they bust (inclusive)
    
    Return:
        Tuple containing the actions choosen by the player, 
        the sum of the player's hand when that action was chosen, and
        a mask indicating which hands in the simulation the player is not busted (i.e. above max value)
        (1 indicating the player can still play, 0 indicated they busted)

        All tensors returned have the shape: (num_sims , num_draws)
    """
    
    cur_sum, dealer_sum = init_state_space

    player_hands = torch.cat((torch.full((partial_sums.shape[0],1), cur_sum , dtype=torch.int32)  , partial_sums + cur_sum), 1)

    # Player hand may excede allowable value --> ensure we know when they bust and game terminates
    # Cannot index directly without doing this otherwise we index non-existent values
    not_busted_filter = torch.where( player_hands <= max_hand_value , 1, 0).int()

    # Create the (2, num_sims , num_draws) index 
    unfilter_indices = torch.stack((player_hands -2 , torch.full(player_hands.shape, dealer_sum) - 1 ) , axis=0).int()  # Indexing starts at zero and player hand sum of 2 is index 0 and dealer hand 1 is index 0
    indices = unfilter_indices * not_busted_filter
    decision_probs  = policy_probabilities[indices[0] , indices[1]]

    # Sample action space
    actions = torch.bernoulli(input = decision_probs).int() # Hit = 1 with probability p 

    return (actions, player_hands, not_busted_filter)
