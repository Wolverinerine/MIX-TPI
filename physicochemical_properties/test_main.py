

"""
    target: test padded_dataset_generator ----- create x array and y label array
"""

""" Scenario for neural network with interaction map input. """

import argparse
import logging
import pandas as pd
import sys
sys.path.append("/mnt/tcr-peptide-prediction-PU")

from physicochemical_properties.feature_builder import CombinedPeptideFeatureBuilder
from physicochemical_properties.peptide_feature import parse_features, parse_operator
from physicochemical_properties.padded_dataset_generator import padded_dataset_generator

def create_parser():
    parser = argparse.ArgumentParser(
        description="Script to train a combined inputs (interaction map) CDR3-epitpe prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_path",
        dest="data_path",
        type=str,
        help="Input csv dataset, as supplied by preprocess_vdjdb script.",
        default="aa_feat/vdjdb-human-tra-trb-mhci-no10x-size.csv"
    )

    parser.add_argument(
        "--features",
        dest="features",
        type=str,
        help="A string of comma separated values listed in peptide_feature.featuresMap.",
        default="hydrophob,isoelectric,mass,hydrophil,charge",
    )
    parser.add_argument(
        "--operator",
        dest="operator",
        type=str,
        choices=["prod", "diff", "absdiff", "layer", "best"],
        help="Can be any of: prod, diff, absdiff, layer or best.",
        default="absdiff",
    )
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # parse cli arguments
    args = create_parser()

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d: %(message)s",
    )

    # log arguments that were used
    for arg, value in sorted(vars(args).items()):
        logging.info("argument %s: %r", arg, value)

    # read (positive) data
    data_source = pd.read_csv(args.data_path, names=['seq'])
    data_source["y"] = 1
    train = data_source

    # get list of features and operator based on input arguments
    features_list = parse_features(args.features)
    print(features_list)
    operator = parse_operator(args.operator)
    feature_builder = CombinedPeptideFeatureBuilder(features_list, operator)

    logging.info("features: " + str(features_list))
    logging.info("operator: " + str(operator))

    train_data = padded_dataset_generator(
        df=train,
        feature_builder=feature_builder
    )

    print(train_data)

    # get length of train dataset
    train_length = len(train) if not neg_shuffle else len(train) * 2

    # shuffle and batch train data
    train_data = train_data.shuffle(
        # buffer equals size of dataset, because positives and negatives are grouped
        buffer_size=train_length,
        seed=42,
        # reshuffle to make each epoch see a different order of examples
        reshuffle_each_iteration=True,
    ).batch(args.batch_size)