import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import shap
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, log_loss, f1_score
from sklearn.preprocessing import OneHotEncoder
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

def get_weights(y, strategy="sqrt_inverse"):
    """
    strategy : "sqrt_inverse" | "inverse" | "none"
    """
    if strategy == "none":
        return np.ones(len(y))
    
    class_counts = y.value_counts().sort_index()
    
    if strategy == "inverse":
        class_weights = 1 / class_counts
    elif strategy == "sqrt_inverse":
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

def evaluate_baseline(X_train, y_train, X_val, y_val, X_test, y_test, n_classes):
    """
    Train a default XGBoost with no tuning and no sample weights.
    Returns log_loss and macro F1 on the test set.
    """
    params = {
        "objective": "multi:softprob",
        "eval_metric": "mlogloss",
        "num_class": n_classes,
        "tree_method": "hist",
        "learning_rate": 0.1,
        "n_estimators": 1000,
        "early_stopping_rounds": 50,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    preds = model.predict(X_test)
    proba = model.predict_proba(X_test)
    
    return {
        "log_loss": log_loss(y_test, proba),
        "f1_macro": f1_score(y_test, preds, average="macro"),
        "f1_per_class": f1_score(y_test, preds, average=None).tolist()
    }

def evaluate_null_model(X_train, y_train, X_test, y_test):
    """
    Majority-class dummy classifier as a floor baseline.
    """
    dummy = DummyClassifier(strategy="most_frequent")
    dummy.fit(X_train, y_train)

    preds = dummy.predict(X_test)
    proba = dummy.predict_proba(X_test)

    return {
        "log_loss": log_loss(y_test, proba),
        "f1_macro": f1_score(y_test, preds, average="macro"),
    }

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
    study.optimize(objective, n_trials=n_trials, n_jobs = 4)

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

def compare_weight_strategies(X_train, y_train, X_val, y_val, X_test, y_test,
                               n_classes, n_trials=50):
    """
    Tune and evaluate each weight strategy by val log loss,
    then return the best strategy and its results.
    """
    from sklearn.metrics import f1_score

    strategies = ["sqrt_inverse", "inverse"]
    strategy_results = {}

    for strategy in strategies:
        weights = get_weights(y_train, strategy=strategy)
        best_params, best_val_loss = tune_xgb(
            X_train, y_train, X_val, y_val,
            weights, n_classes, n_trials
        )
        model = train_xgb(
            X_train, y_train, X_val, y_val,
            weights, best_params, n_classes
        )

        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)

        strategy_results[strategy] = {
            "model": model,
            "best_params": best_params,
            "val_log_loss": best_val_loss,           # from tuning
            "test_log_loss": log_loss(y_test, proba),
            "f1_macro": f1_score(y_test, preds, average="macro"),
            "f1_per_class": f1_score(y_test, preds, average=None).tolist()
        }

    best_strategy = min(strategy_results, key=lambda s: strategy_results[s]["test_log_loss"])

    return strategy_results, best_strategy

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

FEATURE_LABELS = {
    'subject_age': 'Subject Age',
    'moving': 'Moving Violation',
    'mech_nonmoving': 'Mechanical/Non-Moving Violation',
    'dui': 'DUI',
    'collision': 'Collision',
    'motor_assist': 'Motorist Assist',
    'mpc': 'Municipal Police Code',
    'bolo': 'BOLO',
    'subject_race_asian/pacific islander': 'Race: Asian/Pacific Islander',
    'subject_race_black': 'Race: Black',
    'subject_race_hispanic': 'Race: Hispanic',
    'subject_race_other': 'Race: Other',
    'subject_race_white': 'Race: White',
    'subject_sex_female': 'Sex: Female',
    'subject_sex_male': 'Sex: Male',
    'light_condition_dawn': 'Light: Dawn',
    'light_condition_day': 'Light: Day',
    'light_condition_dusk': 'Light: Dusk',
    'light_condition_night': 'Light: Night',
    'district_BAYVIEW': 'District: Bayview',
    'district_CENTRAL': 'District: Central',
    'district_INGLESIDE': 'District: Ingleside',
    'district_MISSION': 'District: Mission',
    'district_NORTHERN': 'District: Northern',
    'district_PARK': 'District: Park',
    'district_RICHMOND': 'District: Richmond',
    'district_SOUTHERN': 'District: Southern',
    'district_TARAVAL': 'District: Taraval',
    'district_TENDERLOIN': 'District: Tenderloin',
    'epc_class_High': 'EPC Class: High',
    'epc_class_Higher': 'EPC Class: Higher',
    'epc_class_Highest': 'EPC Class: Highest',
    'epc_class_Non-EPC': 'EPC Class: Non-EPC',
    **{f'month_{i}': f'Month: {pd.Timestamp(2024, i, 1).strftime("%B")}' for i in range(1, 13)},
    **{f'day_of_week_{d}': f'Day: {d}' for d in ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']}
}

OUTCOME_FILENAMES = {0: "arrests", 1: "citations", 2: "warnings", 3: "no_action"}

def plot_shap(shap_values, X_test, feature_names, period_label):
    """
    Generate a SHAP beeswarm summary plot for each outcome class.

    shap_values     : list of np.ndarray, one array per class (n_samples x n_features)
    feature_names   : list of str
    period_label    : str, used in plot titles
    """
    display_names = [FEATURE_LABELS.get(f, f) for f in feature_names]
    period_slug = "prepolicy" if period_label == "Pre-Policy" else "postpolicy"
    
    for cls_idx, cls_shap in enumerate(shap_values):
        outcome_name = OUTCOME_LABELS.get(cls_idx, f"Class {cls_idx}")
        outcome_slug = OUTCOME_FILENAMES.get(cls_idx, f"class_{cls_idx}")

        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            cls_shap,
            features=X_test,
            feature_names=display_names,
            plot_type="dot",
            max_display=7,
            show=False
        )
        plt.title(f"{period_label} Feature Importance for {outcome_name}", fontsize=13)
        plt.tight_layout()
        plt.savefig(f"../output/{outcome_slug}_shap_{period_slug}.png", dpi=150, bbox_inches="tight")
        plt.show()

OUTCOME_LABELS = {0: "Arrest", 1: "Citation", 2: "Warning", 3: "No Action"}

def report_results(results):
    """
    Print summary metrics and generate SHAP plots for pre/post policy results.

    results : dict
        Output from run_model_pipeline, with keys "pre" and "post".
    """
    for period in ["pre", "post"]:
        r = results[period]
        period_label = "Pre-Policy" if period == "pre" else "Post-Policy"

        print("=" * 60)
        print(f"  {period_label}")
        print("=" * 60)

        # --- Null model ---
        print("\n[Null Model - Majority Class]")
        print(f"  Log Loss : {r['null_model']['log_loss']:.4f}")
        print(f"  F1 Macro : {r['null_model']['f1_macro']:.4f}")

        # --- Untuned XGBoost, no weights ---
        print("\n[Baseline - No Tuning, No Weights]")
        print(f"  Log Loss : {r['baseline']['log_loss']:.4f}")
        print(f"  F1 Macro : {r['baseline']['f1_macro']:.4f}")
        print(f"  F1 Per Class:")
        for cls_idx, f1 in enumerate(r['baseline']['f1_per_class']):
            print(f"    {OUTCOME_LABELS[cls_idx]:<12}: {f1:.4f}")

        # --- Final tuned model ---
        print(f"\n[Final Model - Best Strategy: {r['best_strategy']}]")
        print(f"  Optimal Hyperparameters:")
        for k, v in r['best_params'].items():
            print(f"    {k}: {v}")
        print(f"  Log Loss : {r['test_log_loss']:.4f}")
        print(f"  F1 Macro : {r['f1_macro']:.4f}")
        print(f"  F1 Per Class:")
        for cls_idx, f1 in enumerate(r['f1_per_class']):
            print(f"    {OUTCOME_LABELS[cls_idx]:<12}: {f1:.4f}")

        # --- All weight strategies compared ---
        print(f"\n[Weight Strategy Comparison]")
        for strategy, sr in r['strategy_results'].items():
            print(f"  {strategy:<15} | Val Loss: {sr['val_log_loss']:.4f} "
                  f"| Test Loss: {sr['test_log_loss']:.4f} "
                  f"| F1 Macro: {sr['f1_macro']:.4f}")

        # --- SHAP plots per outcome ---
        print(f"\n[SHAP Plots — {period_label}]")
        plot_shap(r['shap_values'], r['X_test'], r['shap_feature_names'], period_label)

    print("=" * 60)



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

        feature_names = X_train.columns.tolist()

        # classes
        n_classes = len(np.unique(y_train))

        null_metrics = evaluate_null_model(X_train, y_train, X_test, y_test)

        # get baseline model results
        baseline_metrics = evaluate_baseline(
            X_train, y_train, X_val, y_val, X_test, y_test, n_classes
        )

        # tune
        strategy_results, best_strategy = compare_weight_strategies(
            X_train, y_train, X_val, y_val, X_test, y_test,
            n_classes, config["n_trials"]
        )

        best = strategy_results[best_strategy]

        shap_vals = compute_shap(best["model"], X_test, n_classes)

        results[label] = {
            "null_model": null_metrics,
            "shap_feature_names": feature_names,
            "X_test": X_test,
            "baseline": baseline_metrics,
            "strategy_results": strategy_results,
            "best_strategy": best_strategy,
            "model": best["model"],
            "best_params": best["best_params"],
            "test_log_loss": best["test_log_loss"],
            "f1_macro": best["f1_macro"],
            "f1_per_class": best["f1_per_class"],
            "shap_values": shap_vals,
            "confusion_matrix": confusion_matrix(y_test, best["model"].predict(X_test))
        }

    return results