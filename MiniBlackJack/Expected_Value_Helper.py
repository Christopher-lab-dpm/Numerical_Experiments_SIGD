import torch
from typing import Any

def calculate_expected_reward(current_hand_sum: int,  
                              dealer_visible_card: int , 
                              total_dealer_hand: int , 
                              expected_value_array: torch.Tensor, 
                              policy: torch.Tensor,
                              max_hand_value: int= 10 , 
                              max_card_value: int = 5):
  """
  Recursive function that calculates the EXACT expected reward 
  with the current hand and decision policy. Save the values 
  into expected_value_array (from current hand sum and all possible other states which can be can be reached)

  Input:
    current_hand_sum: Integer indicting the total sum of the player's cards  
    dealer_visible_card: The card visible to the player 
    total_dealer_hand: The total value of dealer's visible and unseen cards 
    expected_value_array: torch tensor which records the expected value of 
                         all future possible hands from the initial starting hand. 
                         Has shape: (9,) when for min hand= 2 and max hand = 10
    policy: torch tensor with same dimension as the state space. Contains the probabilities of hitting.
    max_hand_value: int= 10, 
    max_card_value: int = 5
  """
  if current_hand_sum > max_hand_value: # We hit too many times and busted
      return -1

  cur_hand_sum_index= current_hand_sum -2
  if(not expected_value_array[cur_hand_sum_index].isnan()):
    return  expected_value_array[cur_hand_sum_index].item()


  current_reward = None
  if current_hand_sum > total_dealer_hand:
      current_reward = 1
  elif current_hand_sum < total_dealer_hand:
      current_reward = -1
  else: #current_hand_sum  == total_dealer_hand
    current_reward = 0


  hit_prob = policy[cur_hand_sum_index, dealer_visible_card-1].item() # -1 from dealer visible card because indexing starts at 0

  expected_value_array[cur_hand_sum_index] = current_reward * (1-hit_prob)
  for card in range(1,max_card_value+1):
      expected_value_array[cur_hand_sum_index] +=  calculate_expected_reward(current_hand_sum+card, dealer_visible_card ,total_dealer_hand, expected_value_array, policy, max_hand_value, max_card_value) * (1/max_card_value) * hit_prob

  return expected_value_array[cur_hand_sum_index].item()







def expected_val_deriv(current_hand_sum: int,  
                       dealer_visible_card: int , 
                       total_dealer_hand: int, 
                       policy: torch.Tensor, 
                       expected_value_array: torch.Tensor, 
                       deriv_array: torch.Tensor, 
                       accumulated_probs: float = 1.0, 
                       max_hand_value: int = 10 , 
                       max_card_value: int = 5):
  """
  This recursive function calulates the exact derivative of the expected value function of the policy.

  Assumption:
    This methods assumes the probabilities came from: sigmoid(x)


  Recursive function that calculates the EXACT derivative of the expected value function of the policy 
  with the current hand. Save the values into deriv_array.

  Input:
    current_hand_sum: Integer indicting the total sum of the player's cards  
    dealer_visible_card: The card visible to the player 
    total_dealer_hand: The total value of dealer's visible and unseen cards 
    expected_value_array: torch tensor which is prefilled with the expected value of 
                         all future possible hands from the initial starting hand. 
                         Has shape: (9,) when for min hand= 2 and max hand = 10.
                         Is assumed to have come from calculate_expected_reward() function.
    policy: torch tensor with same dimension as the state space. Contains the probabilities of hitting.
    deriv_array: torch.Tensor with same shape as expected_value_array and will be filled with the derivatives.
    accumulated_probs: Records the total porbability of a path during the recursive algorithm, 
    max_hand_value: int= 10, 
    max_card_value: int = 5
  """
  
  if current_hand_sum > max_hand_value: # We are not in the decision space of the agent anymore
      return

  cur_hand_sum_index = current_hand_sum - 2
  dealer_index = dealer_visible_card - 1

  current_stay_reward = None
  if current_hand_sum > total_dealer_hand:
      current_stay_reward = 1
  elif current_hand_sum < total_dealer_hand:
      current_stay_reward = -1
  else: # current_hand_sum  == total_dealer_hand
      current_stay_reward = 0

  hit_prob = policy[cur_hand_sum_index, dealer_index].item()
  E = expected_value_array[cur_hand_sum_index].item() # Total expected value for this hand (assuming it was calculated with the same dealer hand as the current function)

  if(not(hit_prob == 0)):
      current_hit_reward = (E - ((1-hit_prob) * current_stay_reward))/(hit_prob)
  else:
      current_hit_reward = 0

  deriv_array[cur_hand_sum_index] += (current_hit_reward * hit_prob * (1-hit_prob) -1 * current_stay_reward * hit_prob * (1-hit_prob)) * accumulated_probs
  for card in range(1,max_card_value+1):
      expected_val_deriv(current_hand_sum+card,  dealer_visible_card , total_dealer_hand, policy, expected_value_array, deriv_array, accumulated_probs*hit_prob*(1/max_card_value) ,  max_hand_value, max_card_value)