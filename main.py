import argparse
from lib.util import timeit
from lib.automl import AutoML


@timeit
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['classification', 'regression'])
    parser.add_argument('--model-dir')
    parser.add_argument('--train-csv')
    parser.add_argument('--test-csv')
    parser.add_argument('--prediction-csv')
    args = parser.parse_args()

    automl = AutoML(args.model_dir)

    if args.train_csv is not None:
        automl.train(args.train_csv, args.mode)
        automl.save()
    elif args.test_csv is not None:
        automl.load()
        automl.predict(args.test_csv, args.prediction_csv)
    else:
        exit(1)


if __name__ == '__main__':
    main()
