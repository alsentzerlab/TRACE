import argparse
import ast
import re
# pip
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, average_precision_score, roc_auc_score
from tqdm import tqdm
import xgboost as xgb



class Classifier:
    def __init__(self, INPUT_PREFIX, OUTPUT_PREFIX, MODEL=None, RANDOM_SEED=42):
        
        self.input_prefix = INPUT_PREFIX
        self.output_prefix = OUTPUT_PREFIX
        self.random_seed = RANDOM_SEED

        # load data
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None

        # hyper param tuning
        if MODEL != None:
            print("Loading model...")
            self.best_params = pd.read_csv(MODEL)['best_params'].item()
            self.best_params = re.sub(r"np\.(int64|float64)\(([^)]+)\)", r"\2", self.best_params)
            self.best_params = ast.literal_eval(self.best_params)
        else:
            self.best_params  = None
        self.best_ap = None
        self.best_model = None
        self.best_thresh = None
        

    def load_dataset(self):
        emb_file = f'{self.input_prefix}_embeddings.npy'
        var_file = f'{self.input_prefix}_variables.npy'

        X = np.load(emb_file)
        vars_arr = np.load(var_file).astype(int).astype(bool).astype(int)
        y = (vars_arr)
        X = pd.DataFrame(X)
        y = pd.Series(y.squeeze())


        print("Final dataset shape:", X.shape, y.shape)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y,
            test_size=0.15,  
            stratify=y,
            random_state=self.random_seed,
        )
        print("Train:", self.X_train.shape, "Pos rate:", self.y_train.mean())
        print("Test:", self.X_test.shape, "Pos rate:", self.y_test.mean())

        np.save(f'{self.output_prefix}_indices.npy', self.X_test.index.values)
    
    def get_recall_threshold(self):
        print(self.best_params)
        pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        scale_pos_weight = pos_weight * 0.5
        self.best_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=self.best_params['subsample'],
            colsample_bytree=self.best_params['colsample_bytree'],
            max_depth=self.best_params['max_depth'],
            learning_rate=self.best_params['learning_rate'],
            n_estimators=self.best_params['n_estimators'],
            min_child_weight=self.best_params['min_child_weight'],
            gamma=self.best_params['gamma'],
            reg_lambda=self.best_params['reg_lambda'],
            reg_alpha=self.best_params['reg_alpha'],
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            device='cuda'
        )

        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.20,
            stratify=self.y_train,
            random_state=self.random_seed
        )

        self.best_model.fit(X_train, y_train, verbose=False)
        
        # Threshold tuning
        val_probs = self.best_model.predict_proba(X_val)[:, 1]
        
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        precision = precision[:-1]
        recall = recall[:-1]
        valid_idxs = recall >= 0
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        # Apply recall constraint

        best_idx = np.argmax(f1_scores[valid_idxs])

        print(thresholds[valid_idxs][best_idx], f1_scores[valid_idxs][best_idx], recall[valid_idxs][best_idx] )
        # get validation split

    def hyperparameter_tune(self):
        # Stage 1: Coarse search on subset
        # Stage 2: Fine-tune on full data
        
        # ========== STAGE 1: COARSE SEARCH ON SUBSET ==========
        print("="*60)
        print("STAGE 1: Coarse hyperparameter search on data subset")
        print("="*60)
        
        subset_size = 0.25
        X_subset, _, y_subset, _ = train_test_split(
            self.X_train, self.y_train,
            train_size=subset_size,
            stratify=self.y_train,
            random_state=self.random_seed
        )
        X_subset_train, X_subset_test, y_subset_train, y_subset_test = train_test_split(
            X_subset, y_subset,
            train_size=subset_size,
            stratify=y_subset,
            random_state=self.random_seed
        )
        print(f"Subset size: {len(X_subset)} samples ({subset_size*100}% of training data)")
        
        # Coarse parameter grid
        param_grid_coarse = {
            'max_depth': [5, 6, 7, 8, 9],
            'learning_rate': [0.01, 0.02, 0.05, 0.1],
            'n_estimators': [200, 300, 400], 
            'min_child_weight': [5, 10, 15, 20],
            'gamma': [0.0, 0.3, 0.7, 1.0],
            'subsample': [0.5, 0.6, 0.7, 0.8],
            'colsample_bytree': [0.5, 0.6, 0.7, 0.8],
            'reg_lambda': [1.0, 2.0, 4.0, 6.0],
            'reg_alpha': [0.0, 0.5, 1.0, 2.0]
        }
        
        pos_weight = (self.y_train == 0).sum() / (self.y_train == 1).sum()
        scale_pos_weight = pos_weight * 0.5
        
        
        n_iter_coarse = 50  
        best_ap_coarse = -1
        best_params_coarse = None
        
        print(f"Testing {n_iter_coarse} random parameter combinations...\n")
        
        for iteration in tqdm(range(n_iter_coarse)):
            params = {
                'max_depth': np.random.choice(param_grid_coarse['max_depth']),
                'learning_rate': np.random.choice(param_grid_coarse['learning_rate']),
                'n_estimators': np.random.choice(param_grid_coarse['n_estimators']),
                'min_child_weight': np.random.choice(param_grid_coarse['min_child_weight']),
                'gamma': np.random.choice(param_grid_coarse['gamma']),
                'subsample': np.random.choice(param_grid_coarse['subsample']),
                'colsample_bytree': np.random.choice(param_grid_coarse['colsample_bytree']),
                'reg_lambda': np.random.choice(param_grid_coarse['reg_lambda']),
                'reg_alpha': np.random.choice(param_grid_coarse['reg_alpha'])
            }
            
            
            model = xgb.XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                subsample=params['subsample'],
                colsample_bytree=params['colsample_bytree'],
                max_depth=params['max_depth'],
                learning_rate=params['learning_rate'],
                n_estimators=params['n_estimators'],
                min_child_weight=params['min_child_weight'],
                gamma=params['gamma'],
                reg_lambda=params['reg_lambda'],
                reg_alpha=params['reg_alpha'],
                scale_pos_weight=scale_pos_weight,
                early_stopping_rounds=30, 
                random_state=self.random_seed,
                device="cuda"
            )
            
            model.fit(
                X_subset_train, y_subset_train,
                eval_set=[(X_subset_test, y_subset_test)],
                verbose=False
            )

            
            val_probs = model.predict_proba(X_subset_test)[:, 1]
            ap = average_precision_score(y_subset_test, val_probs)
            
            if (iteration + 1) % 10 == 0:
                print(f"Iteration {iteration + 1}/{n_iter_coarse}: Best AP = {best_ap_coarse:.4f}")
            
            if ap > best_ap_coarse:
                best_ap_coarse = ap
                best_params_coarse = params.copy()
                print(f"\n*** New best on subset! ***")
                print(f"Params: {params}")
                print(f"CV PR-AUC: {ap:.4f}")
        
        # ========== STAGE 2: FINE-TUNE ON FULL DATA ==========
        print("\n" + "="*60)
        print("STAGE 2: Fine-tuning top candidates on full data")
        print("="*60)
        
        def create_neighborhood(value, options):
            """Get neighboring values from options list"""
            try:
                idx = options.index(value)
                neighbors = []
                if idx > 0:
                    neighbors.append(options[idx - 1])
                neighbors.append(value)
                if idx < len(options) - 1:
                    neighbors.append(options[idx + 1])
                return neighbors
            except ValueError:
                return [value]
        
        param_grid_fine = {
            'max_depth': create_neighborhood(best_params_coarse['max_depth'], 
                                            param_grid_coarse['max_depth']),
            'learning_rate': create_neighborhood(best_params_coarse['learning_rate'], 
                                                param_grid_coarse['learning_rate']),
            'n_estimators': [300, 400, 500, 600],  # More trees on full data
            'min_child_weight': create_neighborhood(best_params_coarse['min_child_weight'], 
                                                    param_grid_coarse['min_child_weight']),
            'gamma': create_neighborhood(best_params_coarse['gamma'], 
                                        param_grid_coarse['gamma']),
            'subsample': create_neighborhood(best_params_coarse['subsample'], 
                                            param_grid_coarse['subsample']),
            'colsample_bytree': create_neighborhood(best_params_coarse['colsample_bytree'], 
                                                param_grid_coarse['colsample_bytree']),
            'reg_lambda': create_neighborhood(best_params_coarse['reg_lambda'], 
                                            param_grid_coarse['reg_lambda']),
            'reg_alpha': create_neighborhood(best_params_coarse['reg_alpha'], 
                                            param_grid_coarse['reg_alpha'])
        }
        
        print(f"Fine-tuning grid around best params from Stage 1:")
        print(f"Best coarse params: {best_params_coarse}")
        print(f"Best coarse AP: {best_ap_coarse:.4f}\n")
        
        # Full 5-fold CV on full data
        skf_full = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
        
        n_iter_fine = 20  # Fewer iterations on full data
        best_ap = -1
        best_params = None
        
        for iteration in tqdm(range(n_iter_fine)):
            params = {
                'max_depth': np.random.choice(param_grid_fine['max_depth']),
                'learning_rate': np.random.choice(param_grid_fine['learning_rate']),
                'n_estimators': np.random.choice(param_grid_fine['n_estimators']),
                'min_child_weight': np.random.choice(param_grid_fine['min_child_weight']),
                'gamma': np.random.choice(param_grid_fine['gamma']),
                'subsample': np.random.choice(param_grid_fine['subsample']),
                'colsample_bytree': np.random.choice(param_grid_fine['colsample_bytree']),
                'reg_lambda': np.random.choice(param_grid_fine['reg_lambda']),
                'reg_alpha': np.random.choice(param_grid_fine['reg_alpha'])
            }
            
            cv_ap_scores = []
            
            for train_idx, val_idx in skf_full.split(self.X_train, self.y_train):
                X_train_fold, X_val_fold = self.X_train[train_idx], self.X_train[val_idx]
                y_train_fold, y_val_fold = self.y_train[train_idx], self.y_train[val_idx]
                
                model = xgb.XGBClassifier(
                    objective="binary:logistic",
                    eval_metric="logloss",
                    tree_method="hist",
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    max_depth=params['max_depth'],
                    learning_rate=params['learning_rate'],
                    n_estimators=params['n_estimators'],
                    min_child_weight=params['min_child_weight'],
                    gamma=params['gamma'],
                    reg_lambda=params['reg_lambda'],
                    reg_alpha=params['reg_alpha'],
                    scale_pos_weight=scale_pos_weight,
                    early_stopping_rounds=50,
                    random_state=self.random_seed,
                    device="cuda"
                    
                )
                
                model.fit(
                    X_train_fold, y_train_fold,
                    eval_set=[(X_val_fold, y_val_fold)],
                    verbose=False
                )
                
                val_probs = model.predict_proba(X_val_fold)[:, 1]
                ap = average_precision_score(y_val_fold, val_probs)
                cv_ap_scores.append(ap)
            
            mean_ap = np.mean(cv_ap_scores)
            std_ap = np.std(cv_ap_scores)
            
            print(f"Fine-tune iteration {iteration + 1}/{n_iter_fine}: AP = {mean_ap:.4f} ± {std_ap:.4f}")
            
            if mean_ap > best_ap:
                best_ap = mean_ap
                best_params = params.copy()
                print(f"  *** New best on full data! ***\n")
        
        X_train, X_val, y_train, y_val = train_test_split(
            self.X_train, self.y_train,
            test_size=0.20,
            stratify=self.y_train,
            random_state=self.random_seed
        )
        # get best model
        self.best_model = xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            subsample=best_params['subsample'],
            colsample_bytree=best_params['colsample_bytree'],
            max_depth=best_params['max_depth'],
            learning_rate=best_params['learning_rate'],
            n_estimators=best_params['n_estimators'],
            min_child_weight=best_params['min_child_weight'],
            gamma=best_params['gamma'],
            reg_lambda=best_params['reg_lambda'],
            reg_alpha=best_params['reg_alpha'],
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_seed,
            device='cuda'
        )
        
        self.best_model.fit(X_train, y_train, verbose=False)
        
        # Threshold tuning
        val_probs = self.best_model.predict_proba(X_val)[:, 1]
        precision, recall, thresholds = precision_recall_curve(y_val, val_probs)
        f1_scores = 2 * precision * recall / (precision + recall + 1e-9)
        best_idx = np.argmax(f1_scores)
        self.best_thresh = thresholds[best_idx]
        self.best_ap = best_ap
        self.best_params = best_params
        
        print(f"\n{'='*60}")
        print(f"FINAL RESULTS")
        print(f"{'='*60}")
        print(f"Best model params: {best_params}")
        print(f"Best CV PR-AUC: {best_ap:.4f}")
        print(f"Best threshold: {self.best_thresh:.4f}")
        print(f"{'='*60}")

    def model_eval(self):
        train_preds = self.best_model.predict(self.X_train)

        train_precision = precision_score(self.y_train, train_preds, zero_division=0)
        train_recall = recall_score(self.y_train, train_preds, zero_division=0)
        train_f1 = f1_score(self.y_train, train_preds, zero_division=0)

        test_probs = self.best_model.predict_proba(self.X_test)[:, 1]
        test_preds = (test_probs >= self.best_thresh).astype(int)

        test_precision = precision_score(self.y_test, test_preds, zero_division=0)
        test_recall = recall_score(self.y_test, test_preds, zero_division=0)
        test_f1 = f1_score(self.y_test, test_preds, zero_division=0)


        print("\n==== TRAIN METRICS ====")
        print("Precision:", train_precision)
        print("Recall:", train_recall)
        print("F1:", train_f1)

        print("==== TEST METRICS ====")
        print("Precision:", test_precision)
        print("Recall:", test_recall)
        print("F1:", test_f1)
        print("PR-AUC:", average_precision_score(self.y_test, test_probs))
        print("ROC-AUC:", roc_auc_score(self.y_test, test_probs))

        # saving outputs
        np.save(f'{self.output_prefix}_test_probabilities.npy', test_probs)
        np.save(f'{self.output_prefix}_y_test.npy', self.y_test)

        metrics = {
            "best_params": self.best_params,
            "train_precision": train_precision,
            "train_recall": train_recall,
            "train_f1": train_f1,
            "test_precision": test_precision,
            "test_recall": test_recall,
            "test_f1": test_f1,
        }

        pd.DataFrame([metrics]).to_csv(f'{self.output_prefix}_metrics.csv', index=False)
    def run(self):
        self.load_dataset()
        self.get_recall_threshold()
        # self.hyperparameter_tune()
        # self.model_eval()
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create and train a classifier for clinical outcome prediction")
    parser.add_argument('--input', type=str, help="Input prefix for patient embeddings")
    parser.add_argument('--model', type=str, help="Path to best model")

    parser.add_argument('--output', type=str, help="Output prefix for model metrics") 
    args = parser.parse_args()
    # INPUT_PREFIX = "/home/cahoon/TRACE_data/mimiciv/patients_embeddings/processed_50_removed"
    # OUTPUT_PREFIX = "/home/cahoon/TRACE_data/mimiciv/outcome_prediction/removed"
    if args.model != None:
        classifier = Classifier(INPUT_PREFIX = args.input, OUTPUT_PREFIX = args.output, MODEL=args.model)
    else:
        classifier = Classifier(INPUT_PREFIX = args.input, OUTPUT_PREFIX = args.output)
    classifier.run()


