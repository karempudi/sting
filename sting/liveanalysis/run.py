# Run the processes during the live analysis from a 
# paramter file instead of using the UI

import argparse
from sting.liveanalysis.processes import ExptRun, start_live_experiment
from sting.utils.param_io import load_params

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--param_file',
                        help='params file in .json, .yaml or .yml format.',
                        required=True)
    parser.add_argument('-s', '--sim', default=False, action='store_true',
                        help='Will run the experiment in simulation mode.')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("Launching live-analysis processes from command line")
    # Create expt run object based on params
    # and pass the object and params to start_live_expt 
    # from the processes.py module
    params = load_params(args.param_file)
    expt_obj = ExptRun(params)
    start_live_experiment(expt_obj, params, sim=args.sim)

if __name__ == "__main__":
    main()