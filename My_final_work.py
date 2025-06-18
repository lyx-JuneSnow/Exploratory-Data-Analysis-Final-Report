# =============================================================================
# 1. 导入所有必要的库
# =============================================================================
import pandas as pd
import numpy as np
import warnings
import time
from sklearn.model_selection import (train_test_split, 
                                     RandomizedSearchCV, 
                                     GridSearchCV)
from sklearn.preprocessing import (StandardScaler, 
                                   OneHotEncoder, 
                                   MinMaxScaler, 
                                   OrdinalEncoder)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.feature_selection import SelectKBest, chi2, f_classif
import lightgbm as lgb
from sklearn.metrics import (classification_report, 
                             confusion_matrix, 
                             roc_auc_score, 
                             precision_recall_curve, 
                             auc)
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from scipy.stats import uniform, randint, loguniform
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

warnings.filterwarnings('ignore')

# =============================================================================
# 2. 定义辅助函数
# =============================================================================
def get_icd9_group(code_str):
    if not isinstance(code_str, str): 
        code_str = str(code_str)
    code_str = code_str.strip().upper()
    if code_str.startswith('V'): return 'V_Code'
    if code_str.startswith('E'): return 'E_Code'
    try:
        main_code_part = code_str.split('.')[0]
        if not main_code_part.isdigit(): return 'Other'
        val = float(main_code_part)
        if 1 <= val <= 139: return 'Infectious_Parasitic'
        if 140 <= val <= 239: return 'Neoplasms'
        if val == 250: return 'Diabetes'
        if 240 <= val <= 279: return 'Endocrine_Metabolic'
        if 280 <= val <= 289: return 'Blood_Disorders'
        if 290 <= val <= 319: return 'Mental_Disorders'
        if 320 <= val <= 389: return 'Nervous_System'
        if 390 <= val <= 459: return 'Circulatory'
        if 460 <= val <= 519: return 'Respiratory'
        if 520 <= val <= 579: return 'Digestive'
        if 580 <= val <= 629: return 'Genitourinary'
        if 630 <= val <= 679: return 'Pregnancy_Complications'
        if 680 <= val <= 709: return 'Skin_Disorders'
        if 710 <= val <= 739: return 'Musculoskeletal'
        if 780 <= val <= 799: return 'Symptoms_IllDefined'
        if 800 <= val <= 999: return 'Injury_Poisoning'
        return 'Other'
    except ValueError:
        return 'Other'

# =============================================================================
# 3. 定义流水线功能函数
# =============================================================================
def load_and_clean_data(file_path):
    print("--- 步骤 1: 加载和初步清洗数据 ---")
    df = pd.read_csv(file_path)
    df.replace('?', np.nan, inplace=True)
    
    cols_to_drop_high_missing = [
        'weight', 
        'payer_code', 
        'medical_specialty', 
        'max_glu_serum', 
        'A1Cresult'
    ]
    df.drop(columns=cols_to_drop_high_missing, inplace=True)
    df.dropna(subset=['race', 'diag_1', 'diag_2', 'diag_3'], inplace=True)
    
    readmission_map = {'<30': 1, '>30': 0, 'NO': 0}
    df['target'] = df['readmitted'].map(readmission_map)
    df.drop('readmitted', axis=1, inplace=True)
    
    expired_hospice_ids = [11, 13, 14, 19, 20, 21]
    df = df[~df['discharge_disposition_id'].isin(expired_hospice_ids)]
    
    print(f"数据加载和初步清洗完成。数据集形状: {df.shape}")
    return df

def engineer_features(X_train, X_test, id_mapping_file_path):
    print(
        f"\n--- 特征工程 (在 {X_train.shape[0]} 训练样本, "
        f"{X_test.shape[0]} 测试样本上) ---"
    )
    X_train_proc, X_test_proc = X_train.copy(), X_test.copy()
    
    admission_type_map, discharge_disposition_map, admission_source_map = {}, {}, {}
    current_map = None
    with open(id_mapping_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line or line == ',':
                current_map = None
                continue
            if line.lower() == "admission_type_id,description":
                current_map = admission_type_map
            elif line.lower() == "discharge_disposition_id,description":
                current_map = discharge_disposition_map
            elif line.lower() == "admission_source_id,description":
                current_map = admission_source_map
            elif current_map is not None:
                parts = line.split(',', 1)
                if len(parts) == 2:
                    current_map[parts[0].strip()] = parts[1].strip()
    
    id_map_dict = {
        'admission_type_id': admission_type_map, 
        'discharge_disposition_id': discharge_disposition_map, 
        'admission_source_id': admission_source_map
    }
    for id_col, id_map in id_map_dict.items():
        if id_col in X_train_proc.columns:
            descr_col = id_col.replace('_id', '_descr')
            X_train_proc[descr_col] = X_train_proc[id_col].astype(str).map(id_map)
            X_test_proc[descr_col] = X_test_proc[id_col].astype(str).map(id_map)
    
    cols_to_drop = [
        'encounter_id', 'patient_nbr', 'admission_type_id', 
        'discharge_disposition_id', 'admission_source_id'
    ]
    cols_in_df = [col for col in cols_to_drop if col in X_train_proc.columns]
    X_train_proc.drop(columns=cols_in_df, inplace=True, errors='ignore')
    X_test_proc.drop(columns=cols_in_df, inplace=True, errors='ignore')
    
    age_mapping = {f'[{i*10}-{(i+1)*10})': i for i in range(10)}
    drug_mapping = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}
    drug_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    for df_ in [X_train_proc, X_test_proc]:
        if 'age' in df_.columns: 
            df_['age'] = df_['age'].map(age_mapping)
        for col in drug_cols:
            if col in df_.columns: 
                df_[col] = df_[col].map(drug_mapping)
        if 'change' in df_.columns: 
            df_['change'] = df_['change'].map({'No': 0, 'Ch': 1})
        if 'diabetesMed' in df_.columns: 
            df_['diabetesMed'] = df_['diabetesMed'].map({'No': 0, 'Yes': 1})
    
    diag_cols = ['diag_1', 'diag_2', 'diag_3']
    for col in diag_cols:
        if col in X_train_proc.columns:
            train_col_name = f'{col}_group'
            X_train_proc[train_col_name] = X_train_proc[col].astype(str).apply(get_icd9_group)
            X_test_proc[train_col_name] = X_test_proc[col].astype(str).apply(get_icd9_group)
    
    diag_cols_in_df = [col for col in diag_cols if col in X_train_proc.columns]
    X_train_proc.drop(columns=diag_cols_in_df, inplace=True, errors='ignore')
    X_test_proc.drop(columns=diag_cols_in_df, inplace=True, errors='ignore')
    
    categorical_cols = X_train_proc.select_dtypes(include='object').columns.tolist()
    if categorical_cols:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_train_ohe = pd.DataFrame(
            encoder.fit_transform(X_train_proc[categorical_cols]), 
            columns=encoder.get_feature_names_out(categorical_cols), 
            index=X_train_proc.index
        )
        X_test_ohe = pd.DataFrame(
            encoder.transform(X_test_proc[categorical_cols]), 
            columns=encoder.get_feature_names_out(categorical_cols), 
            index=X_test_proc.index
        )
        X_train_proc.drop(columns=categorical_cols, inplace=True)
        X_test_proc.drop(columns=categorical_cols, inplace=True)
        X_train_proc = pd.concat([X_train_proc, X_train_ohe], axis=1)
        X_test_proc = pd.concat([X_test_proc, X_test_ohe], axis=1)
    
    scaler = StandardScaler()
    X_train_final = pd.DataFrame(
        scaler.fit_transform(X_train_proc), 
        columns=X_train_proc.columns, 
        index=X_train_proc.index
    )
    X_test_final = pd.DataFrame(
        scaler.transform(X_test_proc), 
        columns=X_test_proc.columns, 
        index=X_test_proc.index
    )
    
    print(f"特征工程完成。最终特征数量: {X_train_final.shape[1]}")
    return X_train_final, X_test_final

def evaluate_model(model, X_test, y_test, model_name="Model"):
    print(f"\n--- {model_name} 在测试集上的评估结果 ---")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    print("\n混淆矩阵:")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm, 
        index=['真实负类(0)', '真实正类(1)'], 
        columns=['预测负类(0)', '预测正类(1)']
    )
    print(cm_df)
    
    print("\n分类报告:")
    target_names = ['未再入院(0)', '再入院(1)']
    report_dict = classification_report(
        y_test, y_pred, target_names=target_names, output_dict=True
    )
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.4f}")
    print("-" * 35)
    
    class_1_report = report_dict['再入院(1)']
    return {
        'Model': model_name, 
        'AUC-PR': pr_auc, 
        'Recall (class 1)': class_1_report['recall'], 
        'Precision (class 1)': class_1_report['precision'], 
        'F1-score (class 1)': class_1_report['f1-score']
    }

# =============================================================================
# 4. 主执行流程
# =============================================================================
if __name__ == "__main__":
    RANDOM_STATE = 42
    N_ITER_SEARCH = 50 
    CV_FOLDS = 5       
    
    start_time = time.time()
    df = load_and_clean_data('diabetic_data.csv')
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    
    all_results = []
    
    # --- 实验分支 A: 使用全部工程化特征 ---
    print("\n\n================ 实验分支 A: 使用全部工程化特征 ================")
    X_train_A, X_test_A = engineer_features(
        X_train_raw, X_test_raw, 'IDS_mapping.csv'
    )
    
    print(
        f"\n--- [实验A] 步骤 A.1: 标准模型超参数调优 "
        f"(n_iter={N_ITER_SEARCH}, cv={CV_FOLDS}) ---"
    )
    
    models_to_tune = {
        "DecisionTree": DecisionTreeClassifier(
            class_weight='balanced', random_state=RANDOM_STATE
        ),
        "RandomForest": RandomForestClassifier(
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1
        ),
        "LogisticRegression": LogisticRegression(
            solver='liblinear', class_weight='balanced',
            random_state=RANDOM_STATE, max_iter=2000
        ),
        "LightGBM": lgb.LGBMClassifier(
            class_weight='balanced', random_state=RANDOM_STATE, 
            n_jobs=-1, verbosity=-1
        )
    }
    param_grids = {
        "DecisionTree": {
            'max_depth': [5, 10, 15, 20, None], 
            'min_samples_split': randint(2, 50), 
            'min_samples_leaf': randint(1, 50), 
            'criterion': ['gini', 'entropy'], 
            'max_features': ['sqrt', 'log2', None]
        },
        "RandomForest": {
            'n_estimators': randint(100, 600), 
            'max_depth': [10, 20, 30, None], 
            'min_samples_split': randint(2, 20), 
            'min_samples_leaf': randint(1, 20), 
            'bootstrap': [True, False], 
            'max_features': ['sqrt', 'log2']
        },
        "LogisticRegression": {
            'penalty': ['l1', 'l2'], 
            'C': loguniform(1e-4, 1e2)
        },
        "LightGBM": {
            'n_estimators': randint(100, 1000), 
            'learning_rate': uniform(0.01, 0.2), 
            'num_leaves': randint(20, 80), 
            'max_depth': [-1, 10, 20, 30], 
            'reg_alpha': uniform(0, 1), 
            'reg_lambda': uniform(0, 1), 
            'colsample_bytree': uniform(0.6, 0.4),
            'subsample': uniform(0.6, 0.4)
        }
    }
    
    best_params_A = {}
    for name, model in models_to_tune.items():
        print(f"\n>>> [实验A] 开始为 {name} 进行随机搜索调优...")
        rs_cv = RandomizedSearchCV(
            estimator=model, 
            param_distributions=param_grids[name], 
            n_iter=N_ITER_SEARCH, 
            cv=CV_FOLDS, 
            scoring='average_precision', 
            random_state=RANDOM_STATE, 
            verbose=0, 
            n_jobs=-1
        )
        rs_cv.fit(X_train_A, y_train)
        best_params_A[name] = rs_cv.best_params_
        print(f"为 {name} 找到的最佳参数: {rs_cv.best_params_}")
        result = evaluate_model(
            rs_cv.best_estimator_, X_test_A, y_test, 
            model_name=f"Tuned {name} (全特征)"
        )
        all_results.append(result)

    # --- [实验A] 步骤 A.2: 对LightGBM进行精细化网格搜索 ---
    if "LightGBM" in best_params_A:
        print("\n--- [实验A] 步骤 A.2: 对LightGBM进行精细化网格搜索 ---")
        lgbm_rs_best = best_params_A["LightGBM"]
        param_grid_lgbm_focused = {
            'learning_rate': [
                lgbm_rs_best['learning_rate'] * 0.8, 
                lgbm_rs_best['learning_rate'], 
                lgbm_rs_best['learning_rate'] * 1.2
            ], 
            'n_estimators': [
                lgbm_rs_best['n_estimators'] - 50, 
                lgbm_rs_best['n_estimators'], 
                lgbm_rs_best['n_estimators'] + 50
            ], 
            'num_leaves': [
                lgbm_rs_best['num_leaves'] - 5, 
                lgbm_rs_best['num_leaves'], 
                lgbm_rs_best['num_leaves'] + 5
            ]
        }
        lgbm_base_params = {
            k:v for k,v in lgbm_rs_best.items() 
            if k not in param_grid_lgbm_focused
        }
        lgbm_for_grid = lgb.LGBMClassifier(
            **lgbm_base_params, 
            class_weight='balanced',
            random_state=RANDOM_STATE, 
            n_jobs=-1, 
            verbosity=-1
        )
        grid_search_lgbm = GridSearchCV(
            estimator=lgbm_for_grid, 
            param_grid=param_grid_lgbm_focused, 
            cv=CV_FOLDS, 
            scoring='average_precision', 
            verbose=1, 
            n_jobs=-1
        )
        grid_search_lgbm.fit(X_train_A, y_train)
        print(f"网格搜索找到的最终LGBM参数: {grid_search_lgbm.best_params_}")
        gs_lgbm_result = evaluate_model(
            grid_search_lgbm.best_estimator_, X_test_A, y_test, 
            model_name="GridSearched LightGBM (全特征)"
        )
        all_results.append(gs_lgbm_result)

    # --- [实验A] 步骤 A.3: 探索高级特征转换、学习技术 ---
    print("\n--- [实验A] 步骤 A.3: 探索高级特征转换、学习技术 ---")
    
    # 实验 A.3.1: PCA
    pca_pipeline = Pipeline([
        ('pca', PCA(random_state=RANDOM_STATE)), 
        ('classifier', LogisticRegression(
            solver='liblinear', class_weight='balanced', 
            random_state=RANDOM_STATE, max_iter=2000))
    ])
    pca_param_grid = {
        'pca__n_components': randint(20, 81), 
        'classifier__C': loguniform(1e-4, 1e2), 
        'classifier__penalty': ['l1', 'l2']
    }
    rs_cv_pca = RandomizedSearchCV(
        pca_pipeline, pca_param_grid, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, 
        scoring='average_precision', random_state=RANDOM_STATE, 
        verbose=0, n_jobs=-1
    )
    rs_cv_pca.fit(X_train_A, y_train)
    print(f"\nPCA流水线找到的最佳参数: {rs_cv_pca.best_params_}")
    pca_result = evaluate_model(
        rs_cv_pca.best_estimator_, X_test_A, y_test, 
        model_name="Tuned PCA + LR (全特征)"
    )
    all_results.append(pca_result)
    
    # 实验 A.3.2: LDA
    lda_pipeline = Pipeline([
        ('lda', LDA(n_components=1)), 
        ('classifier', LogisticRegression(
            solver='liblinear', class_weight='balanced', 
            random_state=RANDOM_STATE, max_iter=2000))
    ])
    lda_param_grid = {
        'classifier__C': loguniform(1e-4, 1e2), 
        'classifier__penalty': ['l1', 'l2']
    }
    rs_cv_lda = RandomizedSearchCV(
        lda_pipeline, lda_param_grid, n_iter=10, cv=CV_FOLDS, 
        scoring='average_precision', random_state=RANDOM_STATE, 
        verbose=0, n_jobs=-1
    )
    rs_cv_lda.fit(X_train_A, y_train)
    print(f"\nLDA流水线找到的最佳参数: {rs_cv_lda.best_params_}")
    lda_result = evaluate_model(
        rs_cv_lda.best_estimator_, X_test_A, y_test, 
        model_name="Tuned LDA + LR (全特征)"
    )
    all_results.append(lda_result)

    # 实验 A.3.3: RBM
    rbm_pipeline = Pipeline([
        ('minmax', MinMaxScaler()), 
        ('rbm', BernoulliRBM(random_state=RANDOM_STATE)), 
        ('classifier', LogisticRegression(
            solver='liblinear', class_weight='balanced', 
            random_state=RANDOM_STATE, max_iter=2000))
    ])
    rbm_param_grid = {
        'rbm__n_components': randint(50, 151), 
        'rbm__learning_rate': loguniform(1e-3, 1e-1), 
        'rbm__n_iter': [20, 40], 
        'classifier__C': loguniform(1e-2, 1e2)
    }
    rs_cv_rbm = RandomizedSearchCV(
        rbm_pipeline, rbm_param_grid, n_iter=N_ITER_SEARCH, cv=CV_FOLDS, 
        scoring='average_precision', random_state=RANDOM_STATE, 
        verbose=0, n_jobs=-1
    )
    rs_cv_rbm.fit(X_train_A, y_train)
    print(f"\nRBM流水线找到的最佳参数: {rs_cv_rbm.best_params_}")
    rbm_result = evaluate_model(
        rs_cv_rbm.best_estimator_, X_test_A, y_test, 
        model_name="Tuned RBM + LR (全特征)"
    )
    all_results.append(rbm_result)

    # --- 实验分支 B: 基于统计检验进行特征选择并独立调优 ---
    print("\n\n================ 实验分支 B: 基于统计检验的特征选择 ================")
    
    k_proportions_to_test = [0.5, 0.7, 0.9]
    
    for p in k_proportions_to_test:
        print(f"\n>>> [实验B] 测试保留前 {p*100:.0f}% 的统计相关特征...")
        
        # 筛选特征
        cat_cols = X_train_raw.select_dtypes(include='object').columns
        num_cols = X_train_raw.select_dtypes(include=np.number).columns
        
        temp_encoder = OrdinalEncoder(
            handle_unknown='use_encoded_value', unknown_value=-1
        )
        X_train_cat_enc = temp_encoder.fit_transform(X_train_raw[cat_cols])
        
        k_cat = int(len(cat_cols) * p)
        chi2_selector = SelectKBest(chi2, k=k_cat)
        chi2_selector.fit(X_train_cat_enc, y_train)
        selected_cat_cols = cat_cols[chi2_selector.get_support()]
        
        k_num = int(len(num_cols) * p)
        f_selector = SelectKBest(f_classif, k=k_num)
        f_selector.fit(X_train_raw[num_cols], y_train)
        selected_num_cols = num_cols[f_selector.get_support()]
        
        final_selected_cols = selected_num_cols.tolist() + selected_cat_cols.tolist()
        print(f"筛选后特征总数: {len(final_selected_cols)}")
        
        X_train_B_raw = X_train_raw[final_selected_cols]
        X_test_B_raw = X_test_raw[final_selected_cols]

        X_train_B, X_test_B = engineer_features(
            X_train_B_raw, X_test_B_raw, 'IDS_mapping.csv'
        )
        
        print(f"\n>>> [实验B - {p*100:.0f}% 特征] 独立进行超参数调优...")
        for name, model in models_to_tune.items():
            print(f"\n>>> [实验B] 开始为 {name} (在 {p*100:.0f}% 特征上) 进行调优...")
            rs_cv_B = RandomizedSearchCV(
                estimator=model, 
                param_distributions=param_grids[name], 
                n_iter=N_ITER_SEARCH, 
                cv=CV_FOLDS, 
                scoring='average_precision', 
                random_state=RANDOM_STATE, 
                verbose=0, 
                n_jobs=-1
            )
            rs_cv_B.fit(X_train_B, y_train)
            print(
                f"为 {name} (在 {p*100:.0f}% 特征上) 找到的最佳参数: "
                f"{rs_cv_B.best_params_}"
            )
            model_name = f"Tuned {name} ({p*100:.0f}% Stat-Selected)"
            result = evaluate_model(
                rs_cv_B.best_estimator_, X_test_B, y_test, 
                model_name=model_name
            )
            all_results.append(result)

    # --- 最终结果汇总 ---
    print("\n\n=================================================")
    print("           最终模型性能汇总对比")
    print("=================================================")
    results_df = pd.DataFrame(all_results).set_index('Model')
    results_df['PR_Recall_Score'] = (
        0.7 * results_df['AUC-PR'] + 0.3 * results_df['Recall (class 1)']
    )
    results_df.sort_values(
        by='PR_Recall_Score', ascending=False, inplace=True
    )
    
    formatters = {
        'AUC-PR': '{:,.4f}'.format, 
        'Recall (class 1)': '{:,.4f}'.format, 
        'Precision (class 1)': '{:,.4f}'.format,
        'F1-score (class 1)': '{:,.4f}'.format, 
        'PR_Recall_Score': '{:,.4f}'.format
    }
    print(results_df.to_string(formatters=formatters))
    print("=================================================")

    end_time = time.time()
    total_time = end_time - start_time
    print(f"\n流水线总执行时间: {total_time / 60:.2f} 分钟。")