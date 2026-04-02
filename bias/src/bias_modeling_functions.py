import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
import subprocess
import xgboost as xgb

def preprocess_stops(df):
    """
    Clean and transform raw stops data for modeling.

    df : pandas.DataFrame
        Raw dataset containing stop-level information.

    Returns
    pandas.DataFrame
        Processed DataFrame with engineered features and selected columns,
        including outcome variable and temporal features.
    """
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.day_name()
    df['month'] = df['date'].dt.month

    df['outcome'] = np.select(
        [
            df['arrest_made'] == 1,
            df['citation_issued'] == 1,
            df['warning_issued'] == 1
        ],
        [0, 1, 2],
        default=3
    )

    df = df.drop(columns=[
        'warning_issued', 'citation_issued', 'arrest_made',
        'time', 'lat', 'lng', 'pct_over75', 'pct_poc',
        'pct_disab', 'search_conducted', 'search_vehicle'
    ])

    return df

def split_pre_post(df, policy_date):
    """
    Split dataset into pre-policy and post-policy subsets.

    df : pandas.DataFrame
        Processed dataset containing a date column.
    policy_date : str
        Date threshold used to split the dataset (YYYY-MM-DD).

    Returns
    tuple of pandas.DataFrame
        Two DataFrames: (pre_policy_df, post_policy_df), each with
        the date column removed.
    """
    pre = df[df['date'] < policy_date].drop(columns=['date'])
    post = df[df['date'] >= policy_date].drop(columns=['date'])
    return pre, post

def get_weights(y):
    """
    Compute sample weights to balance class frequencies.

    y : pandas.Series
        Outcome variable containing class labels.

    Returns
    numpy.ndarray
        Array of sample weights corresponding to each observation,
        using square-root inverse frequency scaling.
    """
    class_counts = y.value_counts().sort_index()
    class_weights = 1 / np.sqrt(class_counts)
    class_weights = class_weights / class_weights.min()
    return y.map(class_weights.to_dict()).values

def train_xgb(X_train, y_train, X_val, y_val, sample_weights, best_params, n_classes):
    """
    Train final XGBoost model using tuned hyperparameters.

    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training outcome labels.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_val : pandas.Series
        Validation outcome labels.
    sample_weights : numpy.ndarray
        Sample weights for training data.
    best_params : dict
        Tuned hyperparameters from Optuna.
    n_classes : int
        Number of outcome classes.

    Returns
    xgboost.XGBClassifier
        Trained XGBoost model.
    """
    full_params = {
        **best_params,
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": n_classes,
        "tree_method": "hist",
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }

    model = xgb.XGBClassifier(**full_params)

    model.fit(
        X_train, y_train,
        sample_weight=sample_weights,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    return model

def tune_xgb(X_train, y_train, X_val, y_val, sample_weights, n_classes, n_trials=50):
    """
    Tune XGBoost hyperparameters using Optuna.

    X_train : pandas.DataFrame
        Training feature matrix.
    y_train : pandas.Series
        Training outcome labels.
    X_val : pandas.DataFrame
        Validation feature matrix.
    y_val : pandas.Series
        Validation outcome labels.
    sample_weights : numpy.ndarray
        Sample weights for training observations.
    n_classes : int
        Number of outcome classes.
    n_trials : int, optional
        Number of Optuna trials to run. Default is 50.

    Returns
    tuple
        best_params : dict
            Dictionary of optimal hyperparameters.
        best_value : float
            Best validation log loss achieved.
    """
    def objective(trial):
        params = {
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": n_classes,
            "tree_method": "hist",
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 20.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 20.0, log=True),
            "learning_rate": 0.1,
            "n_estimators": 1000,
            "early_stopping_rounds": 50,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            sample_weight=sample_weights,
            verbose=False
        )

        return log_loss(y_val, model.predict_proba(X_val))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_params, study.best_value

def compute_shap(model, X_test, n_classes):
    """
    Compute SHAP values for a trained XGBoost model.

    model : xgboost.XGBClassifier
        Trained model.
    X_test : pandas.DataFrame
        Test feature matrix.
    n_classes : int
        Number of outcome classes.

    Returns
    list of numpy.ndarray
        List of SHAP value arrays, one per class.
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    shap_values_per_class = [
        shap_values[:, :, i] for i in range(n_classes)
    ]

    return shap_values_per_class

def evaluate_model(model, X_test, y_test):
    """
    Evaluate model performance on test data.

    model : xgboost.XGBClassifier
        Trained model.
    X_test : pandas.DataFrame
        Test feature matrix.
    y_test : pandas.Series
        True outcome labels.

    Returns
    tuple
        accuracy : float
            Classification accuracy on test set.
        confusion_matrix : numpy.ndarray
            Confusion matrix of predictions.
    """
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    cm = confusion_matrix(y_test, preds)

    return acc, cm

def strat_split(df, outcome_col='outcome', epc_col='epc_class', 
                                  train_frac=0.75, val_frac=0.15, test_frac=0.15, random_state=42):
    """
    Splits df into train/val/test with:
    Stratified by outcome
    Ensures EPC classes are present in each set
    """
    
    # Initial train / temp split
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_frac),
        stratify=df[outcome_col],
        random_state=random_state
    )
    
    # Split temp into validation and test
    val_df, test_df = train_test_split(
        temp_df,
        # proportion of temp
        test_size=test_frac / (test_frac + val_frac),  
        stratify=temp_df[outcome_col],
        random_state=random_state
    )
    
    splits = {'train': train_df, 'val': val_df, 'test': test_df}
    
    # Ensure each EPC class appears in each split
    all_epc_classes = set(df[epc_col])
    
    for split_name, split_df in splits.items():
        missing_epc = all_epc_classes - set(split_df[epc_col])
        for epc in missing_epc:
            # Find a split that contains this EPC class and take one row
            for other_name, other_df in splits.items():
                if other_name == split_name:
                    continue
                if epc in set(other_df[epc_col]):
                    row_to_move = other_df[other_df[epc_col] == epc].iloc[0]
                    split_df = pd.concat([split_df, pd.DataFrame([row_to_move])], ignore_index=True)
                    splits[other_name] = other_df.drop(row_to_move.name)
                    break
        splits[split_name] = split_df
    
    return splits['train'], splits['val'], splits['test']

def fit_encoder(df, cat_cols, outcome_col):
    """Fit encoder on training data"""
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoder.fit(df[cat_cols])
    return encoder

def encode(df, cat_cols, outcome_col, encoder=None):
    """
    Encode using a pre-fitted encoder
    If no encoder provided, assumes no categorical cols need encoding
    """
    non_cat_cols = [c for c in df.columns if c not in cat_cols + [outcome_col]]
    
    # encode categoricals
    if encoder is not None:
        encoded = encoder.transform(df[cat_cols])
        encoded_cols = encoder.get_feature_names_out(cat_cols)
        encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
    
    # combine with non-categorical features
    X = pd.concat([df[non_cat_cols].reset_index(drop=True), 
                   encoded_df.reset_index(drop=True)], axis=1)
    y = df[outcome_col]
    
    return X, y

def run_model_pipeline(df, config):
    """
    Runs the full modeling pipeline for pre- and post-policy datasets.

    df : pandas.DataFrame
        Raw input dataframe containing stop-level data.
    config : dict
        Configuration dictionary containing:
        - policy_date : str or datetime
        - categorical_cols : list of str
        - outcome_col : str
        - n_trials : int

    Returns
    results : dict
        Dictionary with keys "pre" and "post", each containing:
        - model : trained XGBoost model
        - accuracy : float
        - confusion_matrix : numpy.ndarray
        - shap_values : object
        - best_params : dict
    """

    # preprocess
    df = preprocess_stops(df)

    # split pre/post
    pre_df, post_df = split_pre_post(df, config["policy_date"])

    results = {}

    for label, dataset in [("pre", pre_df), ("post", post_df)]:

        # split
        train, val, test = strat_split(dataset)

        # encode
        encoder = fit_encoder(train, config["categorical_cols"], config["outcome_col"])

        X_train, y_train = encode(train, config["categorical_cols"], config["outcome_col"], encoder)
        X_val, y_val = encode(val, config["categorical_cols"], config["outcome_col"], encoder)
        X_test, y_test = encode(test, config["categorical_cols"], config["outcome_col"], encoder)

        # weights
        weights = get_weights(y_train)

        # classes
        n_classes = len(np.unique(y_train))

        # tune
        best_params, best_loss = tune_xgb(
            X_train, y_train, X_val, y_val,
            weights, n_classes,
            config["n_trials"]
        )

        # train
        model = train_xgb(
            X_train, y_train, X_val, y_val,
            weights, best_params, n_classes
        )

        # evaluate
        acc, cm = evaluate_model(model, X_test, y_test)

        # shap
        shap_vals = compute_shap(model, X_test, n_classes)

        results[label] = {
            "model": model,
            "accuracy": acc,
            "confusion_matrix": cm,
            "shap_values": shap_vals,
            "best_params": best_params
        }

    return results
