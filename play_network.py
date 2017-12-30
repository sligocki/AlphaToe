import argparse

import tensorflow as tf

from games import tic_tac_toe
from common import network_helpers

parser = argparse.ArgumentParser()
parser.add_argument("--network-filename", required=True)
parser.add_argument("--human-plays", choices=("X", "O"), default="O")
args = parser.parse_args()

game_spec = tic_tac_toe.TicTacToeGameSpec()

def load_network_player(network_filename, hidden_layers):
    session = tf.Session()
    input_layer, output_layer, variables = network_helpers.create_network(
        game_spec.board_squares(), hidden_layers)
    network_helpers.load_network(session, variables, network_filename)

    def network_player(board_state, side):
        print
        print "Network player (%s)" % side
        tic_tac_toe.print_game_state(board_state)

        move_probs = network_helpers.get_stochastic_network_move(
            session, input_layer, output_layer, board_state, side, log=True)
        move = game_spec.flat_move_to_tuple(move_probs.argmax())

        print "Network move:", move
        return move
    return network_player


network_player = load_network_player(args.network_filename,
                                     (100, 100, 100))
human_player = tic_tac_toe.manual_player
if args.human_plays == "X":
    x_player = human_player
    o_player = network_player
else:
    o_player = human_player
    x_player = network_player

winner = game_spec.play_game(x_player, o_player)
print "Winner:", tic_tac_toe.player_to_string(winner)
