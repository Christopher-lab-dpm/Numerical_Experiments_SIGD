import torch
from typing import Any

class Agent:
    def __init__(self , 
                 Policy : Any , 
                 state_space_shape : tuple[int, int]):
        
        self.policy = Policy
        self.params = torch.zeros(state_space_shape, dtype=torch.float64)



    def apply_Policy(self, **kargs) -> Any:
        """
        Apply the agent decision making policy

        It should return actions valid in the current state space.
        """
        return self.policy(**kargs)
    
    
    def get_probabilities(self, alternative_params = None):
         """
         If no argument is specified, the probabilities for the current set of parameters are returned.
         Otherwise, new probabilities will be calculated for the new parameters
         """
         if(alternative_params == None):
              return torch.sigmoid(self.params)
         else:
              return torch.sigmoid(alternative_params)


    def update_params(self , new_params : torch.Tensor):
           """
           It is assumed that new params has the correct shape
           """
           self.params = new_params



def calculate_reward(actions: torch.Tensor,
                    player_hands: torch.Tensor,
                    not_busted_filter: torch.Tensor,
                    dealer_total_hand: int,
                    max_hand_value: int) ->  torch.Tensor:
    
    """
    Given the actions taken during a sequence of simualtions of draws and
    a mask indicating if the action was taken after exceeding the max hand limit,
    calculate the reward of the set of actions for each simulation.

    Exceed maximal values (i.e. bust) or fail to beat dealer -> reward -1
    Tie with dealer -> reward = 0
    Beat deal -> reard = 1

    Args:
        actions: Tensor of shape (num_sims , num_draws) indicating 
                    wether the player hit or stays. (Is not filtered, thus may not contain valid actions)

        player_hands: Tensor of shape of shape (num_sims , num_draws). 
                        Includes the total of player hand over the sequence of draws
                        as if the player keep "hitting" even after busting.
        
        not_busted_filter: Binary map of shape (num_sims , num_draws) meant to be used as a mask such that,
                            1 indicated a valid action and player hand and 0
                            means the player busted, thus no action taken is valid
        
        dealer_total_hand: The total sum (seen and unseen card) of the dealer hand

        max_hand_value: Total inclusive allowable value a player can have (the blackjack value)
    
    Output:
        rewards: Torch tensor of shape (num_sims, )
    """

    # The first zero of each row will indicate when we stop that simulated hand
    # A one means we hit.
    # Reasoning --> could have chosen to stop playing when not busted 
    #               but next draws we done before that decision and it
    #               indicate to hit again regardless. 
    #               Thus first zero is the stopping condition here (either bust or not hit).
    hits_until_terminate =  actions * not_busted_filter
    # Find the first valid zero in each row
    terminate_indices = hits_until_terminate.argmin(dim=1) # Tensor of shape (sim_num, )


    final_hand = player_hands[torch.arange(start=0, end=player_hands.shape[0], step=1) , terminate_indices]

    # Calculate the reward of the games
    rewards =  torch.zeros_like(final_hand)
    rewards[final_hand < dealer_total_hand] = -1  # fail_to_beat dealer
    rewards[final_hand > max_hand_value] = -1 # busted
    rewards[(final_hand > dealer_total_hand) & (final_hand <= max_hand_value)] = 1 # Beat the house
    rewards[final_hand == dealer_total_hand] = 0 # It was a tie
    
    # print(dealer_total_hand)
    # print(final_hand)
    return rewards
    