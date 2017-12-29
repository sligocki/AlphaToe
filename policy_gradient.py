"""
Builds and trains a neural network that uses policy gradients to learn to play Tic-Tac-Toe.

The input to the network is a vector with a number for each space on the board. If the space has one of the networks
pieces then the input vector has the value 1. -1 for the opponents space and 0 for no piece.

The output of the network is a also of the size of the board with each number learning the probability that a move in
that space is the best move.

The network plays successive games randomly alternating between going first and second against an opponent that makes
moves by randomly selecting a free space. The neural network does NOT initially have any way of knowing what is or is not
a valid move, so initially it must learn the rules of the game.

I have trained this version with success at 3x3 tic tac toe until it has a success rate in the region of 75% this maybe
as good as it can do, because 3x3 tic-tac-toe is a theoretical draw, so the random opponent will often get lucky and
force a draw.
"""
import argparse
import functools
import random

from common.network_helpers import create_network
from games.tic_tac_toe import TicTacToeGameSpec
from techniques.train_policy_gradient import train_policy_gradients

parser = argparse.ArgumentParser()
parser.add_argument("--opponent", default="random",
                    choices=("random", "perfect", "mixed"),
                    help="type of opponent to train against.")
parser.add_argument("--num-games", type=int, default=1000000,
                    help="Number of games to run before stopping. "
                    "-1 for infinite.")
parser.add_argument("hidden_layers", nargs="*", type=int,
                    help="List of hidden layer sizes")
args = parser.parse_args()

BATCH_SIZE = 100  # every how many games to do a parameter update?
LEARN_RATE = 1e-4
PRINT_RESULTS_EVERY_X = 10000  # every how many games to print the results

# to play a different game change this to another spec, e.g TicTacToeXGameSpec or ConnectXGameSpec, to get these to run
# well may require tuning the hyper parameters a bit
game_spec = TicTacToeGameSpec()

if not args.hidden_layers:
  args.hidden_layers = (100, 100, 100)

# create_network_func = functools.partial(create_network, game_spec.board_squares(), (100, 100, 100))
create_network_func = functools.partial(create_network, game_spec.board_squares(), args.hidden_layers)

network_file_path = 'current_network'
for n in args.hidden_layers:
  network_file_path = network_file_path + ("_%05d" % n)

network_file_path = network_file_path + ".p"

random_opponent = game_spec.get_random_player_func()
perfect_opponent = game_spec.get_perfect_player()
def mixed_opponent(*args, **kwds):
    opponent = random.choice([random_opponent, perfect_opponent])
    return opponent(*args, **kwds)

if args.opponent == "random":
    opponent_func = random_opponent
elif args.opponent == "perfect":
    opponent_func = perfect_opponent
elif args.opponent == "mixed":
    opponent_func = mixed_opponent
else:
    raise Exception, "Invalid value for --opponent"

train_policy_gradients(game_spec, create_network_func, network_file_path,
                       opponent_func=opponent_func,
                       number_of_games=args.num_games,
                       batch_size=BATCH_SIZE,
                       learn_rate=LEARN_RATE,
                       print_results_every=PRINT_RESULTS_EVERY_X)
