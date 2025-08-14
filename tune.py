import optuna
from train import run_training
from itertools import product
import os
import json
import time
import copy
import argparse

def objective(trial, base_args):
    """
    Defines and executes a single Optuna trial.

    This function is called by Optuna for each trial. It suggests hyperparameters,
    runs the training process with those hyperparameters, and returns the
    accuracy, which Optuna will then try to maximize.

    Args:
        trial (optuna.trial.Trial): An Optuna trial object used to suggest
            hyperparameters.
        base_args (argparse.Namespace): A copy of the base command-line arguments.

    Returns:
        float: The validation accuracy of the model for the trial.
    """
    args = copy.deepcopy(base_args)

    # Set dynamic and fixed parameters for the trial
    args.exp_name = "results"

    # Suggest hyperparameters based on the params_to_search list
    if 'lr' in args.params_to_search:
        args.lr = trial.suggest_float('lr', 1e-5, 1e-2, log=True)
    if 'batch_size' in args.params_to_search:
        args.batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32])
    if 'dropout_conv_p' in args.params_to_search:
        args.dropout_conv_p = trial.suggest_float('dropout_conv_p', 0.0, 0.3)
    if 'dropout_fc_p' in args.params_to_search:
        args.dropout_fc_p = trial.suggest_float('dropout_fc_p', 0.0, 0.3)
    if 'weight_decay' in args.params_to_search:
        args.weight_decay = trial.suggest_categorical('weight_decay', [0, 5e-4])
    
    print(f"Trial {trial.number}:")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  CNN Dropout: {args.dropout_conv_p}")
    print(f"  FC Dropout: {args.dropout_fc_p}")
    print(f"  Weight Decay: {args.weight_decay}")
    print(f"  Dataset: {args.dataset}")

    # Train and evaluate the model
    accuracy = run_training(args)
    
    return accuracy

def generate_param_combinations(param_space):
    """
    Generates all possible combinations of hyperparameters.

    Creates a list of dictionaries, where each dictionary represents a unique
    combination of hyperparameters based on the provided parameter space. This
    is used to create an initial set of trials for the Optuna study.

    Args:
        param_space (dict): A dictionary where keys are parameter names and
                           values are lists of possible values for that
                           parameter.

    Returns:
        list: A list of dictionaries, with each dictionary representing one
              combination of hyperparameters.
    """
    param_names = list(param_space.keys())
    param_values = list(param_space.values())
    
    combinations = []
    for values in product(*param_values):
        param_dict = {name: value for name, value in zip(param_names, values)}
        combinations.append(param_dict)
    
    return combinations

def tune(args):
    """
    Performs hyperparameter tuning using Optuna.

    This function sets up and runs an Optuna study to find the best
    hyperparameters for the model. It starts with a grid search over a predefined
    parameter space and then uses a Tree-structured Parzen Estimator (TPE)
    sampler for further Bayesian optimization.

    Args:
        args (argparse.Namespace): Command-line arguments containing tuning
                                   parameters and other configurations.

    Returns:
        dict: A dictionary containing the best hyperparameters found by the study.
    """
    study_name = args.name
    random_trial_coeff = args.random_trial_coeff
    bayesian_trial_coeff = args.bayesian_trial_coeff
    output_dir = args.output_dir
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(output_dir, f'{study_name}_{timestamp}')
    os.makedirs(run_dir, exist_ok=True)
    
    # Set the root for all experiment outputs to be the run directory
    args.exp_root = run_dir

    full_param_space = {
        'lr': [1e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 5e-3, 1e-2],
        'batch_size': [8, 16, 32, 64],
        'dropout_conv_p': [0, 0.1, 0.3],
        'dropout_fc_p': [0, 0.1, 0.3],
        'weight_decay': [0, 5e-4],
    }

    # Filter the parameter space based on the params_to_search argument
    param_space = {key: full_param_space[key] for key in args.params_to_search if key in full_param_space}

    param_combinations = generate_param_combinations(param_space)
    start_trials = int(len(param_combinations) * random_trial_coeff)
    n_trials = int(start_trials * bayesian_trial_coeff)
    print(f'Running {n_trials} trials')
    
    sampler = optuna.samplers.TPESampler(       
        n_startup_trials=start_trials,
        multivariate=True
    )   
    
    storage_path = f"sqlite:///{os.path.join(run_dir, study_name)}.db"
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_path,
        load_if_exists=False,
        direction='maximize',
        sampler=sampler, 
    )

    for params in param_combinations:
        study.enqueue_trial(params)

    study.optimize(lambda trial: objective(trial, args), n_trials=n_trials)

    # Save results
    best_params = study.best_params
    best_value = study.best_value

    # Save best hyperparameters to JSON
    best_params_path = os.path.join(run_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({'best_value': best_value, 'best_params': best_params}, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}")

    # Save summary to a text file
    summary_path = os.path.join(run_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Study: {study_name}\n")
        f.write(f"Best Value (Accuracy): {best_value:.4f}\n")
        f.write("Best Hyperparameters:\n")
        for key, value in best_params.items():
            f.write(f"  {key}: {value}\n")
    print(f"Tuning summary saved to {summary_path}")
    
    print("\n" + "="*50)
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Accuracy): {trial.value:.4f}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("="*50 + "\n")
    
    return study.best_params, study.best_value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SigNet Hyperparameter Tuning Script')

    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=['data/CEDAR', 'sign_data'], default='data/CEDAR', help='Dataset directory path')
    parser.add_argument('--train_file', type=str, default='train_subset_0.1.csv', help='Train data file')
    parser.add_argument('--test_file', type=str, default='test_subset_0.4.csv', help='Test data file')
    parser.add_argument('--img_height', type=int, default=155, help='Image height for resizing')
    parser.add_argument('--img_width', type=int, default=220, help='Image width for resizing')
    
    # Training parameters
    parser.add_argument('--num_epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--log_interval', type=int, default=10, help='Logging interval for training progress')
    parser.add_argument('--seed', type=int, default=2020, help='Random seed for reproducibility')
    
    # Optimizer parameters
    parser.add_argument('--optimizer', type=str, choices=['rmsprop', 'adam', 'sgd'], default='adam', help='Optimizer type')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and validation')
    parser.add_argument('--eps', type=float, default=1e-8, help='Optimizer epsilon (for RMSprop and Adam)')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='Weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum (for RMSprop and SGD)')
    
    # Scheduler parameters
    parser.add_argument('--scheduler_step', type=int, default=5, help='Learning rate scheduler step size')
    parser.add_argument('--scheduler_gamma', type=float, default=0.5, help='Learning rate scheduler gamma')
    
    # Loss function parameters
    parser.add_argument('--alpha', type=float, default=1.0, help='Contrastive loss alpha parameter')
    parser.add_argument('--beta', type=float, default=1.0, help='Contrastive loss beta parameter')
    parser.add_argument('--margin', type=float, default=1.0, help='Contrastive loss margin parameter')
    
    # Dropout parameters
    parser.add_argument('--dropout_conv_p', type=float, default=0.2, help='Dropout probability for convolutional layers')
    parser.add_argument('--dropout_fc_p', type=float, default=0.3, help='Dropout probability for fully connected layers')

    # Model graph logging
    parser.add_argument('--log_model_graph', action='store_true', default=True, help='Log model architecture to TensorBoard')

    # Add tuning-specific arguments
    parser.add_argument('--name', default='signet_tuning', help='The name of the study')
    parser.add_argument('--random_trial_coeff', type=float, default=1.2, help='The coefficient for the number of startup trials')
    parser.add_argument('--bayesian_trial_coeff', type=float, default=1.2, help='The coefficient for the number of trials')
    parser.add_argument('--output_dir', type=str, default='tuning_results', help='Directory to save tuning results')
    parser.add_argument('--params_to_search', nargs='+', default=['lr', 'batch_size', 'dropout_conv_p', 'dropout_fc_p', 'weight_decay'], 
                        help='List of hyperparameters to search (e.g., lr, batch_size)')
    
    args = parser.parse_args()
    
    best_params, best_value = tune(args)
    print("Best hyperparameters found:")
    print(best_params)
    print(f"Best value: {best_value}")

    best_params_path = os.path.join(args.output_dir, 'best_params.json')
    with open(best_params_path, 'w') as f:
        json.dump({'best_value': best_value, 'best_params': best_params}, f, indent=4)
    print(f"Best hyperparameters saved to {best_params_path}") 