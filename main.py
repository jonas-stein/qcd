import os
import evaluation

path = os.path.abspath(os.path.dirname(__file__))


def main():
    evaluation.modularity_deviation(1, 1)  # WORKS
    evaluation.nmi_deviation(1, 1)  # WORKS
    evaluation.com_deviation(1, 1)  # WORKS
    evaluation.separation_node_set_deviation(1, 1)  # WORKS
    # evaluation.prediction_deviation()   # TODO
    evaluation.check_surjective_and_injective_correctness(1, 1)  # WORKS
    evaluation.intra_probability_SBM_plots(1)
    # print(graphs.create_seed_list())


if __name__ == "__main__":
    main()
