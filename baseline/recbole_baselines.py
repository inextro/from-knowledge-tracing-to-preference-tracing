import argparse
from recbole.quick_start import run_recbole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True)

    args = parser.parse_args()
    model_name = args.model_name
    
    if model_name == 'DIEN':
        config = './configs/config_DIEN.yaml'
    else:
        config = './configs/config.yaml'

    run_recbole(model=model_name, dataset='ml', config_file_list=[config])


if __name__ == '__main__':
    main()