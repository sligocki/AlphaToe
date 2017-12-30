import argparse

import tensorflow as tf

from games import tic_tac_toe
from common import network_helpers

parser = argparse.ArgumentParser()
parser.add_argument(
    "-X", default="human",
    metavar="{human,random,perfect,mixed,network:NETWORK_FILENAME}")
parser.add_argument(
    "-O", default="human",
    metavar="{human,random,perfect,mixed,network:NETWORK_FILENAME}")
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

def load_player(player_flag):
    if player_flag == "human":
        return tic_tac_toe.manual_player
    elif player_flag == "random":
        return game_spec.get_random_player_func()
    elif player_flag == "perfect":
        return tic_tac_toe.perfect_player
    elif player_flag == "mixed":
        raise Exception, "Not implemented"
    else:
        assert player_flag.startswith("network:")
        network_filename = player_flag[len("network:"):]
        return load_network_player(network_filename,
                                   (100, 100, 100))

winner = game_spec.play_game(load_player(args.X), load_player(args.O))
print "Winner:", tic_tac_toe.player_to_string(winner)
