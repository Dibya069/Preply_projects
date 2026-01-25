import sys, os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set non-interactive backend before importing pyplot
import matplotlib.pyplot as plt

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_mdoel

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier

@dataclass
class DataModelTrainConfig:
    train_model_path = os.path.join("artifacts", "model.pkl")
    feature_importance_dir = os.path.join("artifacts", "feature_importance")

class ModelTrainig:
    def __init__(self):
        self.model_training_config = DataModelTrainConfig()
        # Feature names for cancer dataset
        self.feature_names = [
            'Age', 'Gender', 'Air Pollution', 'Alcohol use', 'Dust Allergy',
            'OccuPational Hazards', 'Genetic Risk', 'chronic Lung Disease',
            'Balanced Diet', 'Obesity', 'Smoking', 'Passive Smoker',
            'Chest Pain', 'Coughing of Blood', 'Fatigue', 'Weight Loss',
            'Shortness of Breath', 'Wheezing', 'Swallowing Difficulty',
            'Clubbing of Finger Nails', 'Frequent Cold', 'Dry Cough', 'Snoring'
        ]

    def calculate_feature_importance_score(self, model, model_name):
        """
        Calculate a score based on how many features the model uses.
        Higher score means more features are being used (more distributed importance).
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
                logging.info(f"Feature importances: {importance}")

                # Count features with importance > 0.01 (1%)
                significant_features = np.sum(importance > 0.01)

                # Calculate distribution score (using entropy-like measure)
                # More distributed = higher score
                non_zero_importance = importance[importance > 0]
                if len(non_zero_importance) > 0:
                    # Normalize
                    normalized_importance = non_zero_importance / np.sum(non_zero_importance)
                    # Calculate entropy (higher = more distributed)
                    entropy = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
                    ##-(summation of all the avg values) * log(avg values)base10
                else:
                    entropy = 0

                logging.info(f"{model_name} - Significant features: {significant_features}, Entropy: {entropy:.4f}")

                return significant_features, entropy
            else:
                logging.warning(f"{model_name} does not support feature_importances_")
                return 0, 0

        except Exception as e:
            logging.error(f"Error calculating feature importance for {model_name}: {str(e)}")
            return 0, 0

    def save_feature_importance_plot(self, model, model_name):
        """
        Save feature importance plot for a model.
        """
        try:
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_

                # Create DataFrame
                feat_importance = pd.DataFrame({
                    'Feature': self.feature_names,
                    'Importance': importance
                }).sort_values(by='Importance', ascending=False)

                # Create directory if it doesn't exist
                os.makedirs(self.model_training_config.feature_importance_dir, exist_ok=True)

                # Plot top 10 features
                plt.figure(figsize=(10, 6))
                plt.barh(feat_importance['Feature'][:10][::-1],
                        feat_importance['Importance'][:10][::-1],
                        color='skyblue')
                plt.title(f'Feature Importance - {model_name}')
                plt.xlabel('Importance')
                plt.ylabel('Feature')
                plt.tight_layout()

                # Save plot
                plot_path = os.path.join(
                    self.model_training_config.feature_importance_dir,
                    f'{model_name.replace(" ", "_")}_importance.png'
                )
                plt.savefig(plot_path)
                plt.close()

                logging.info(f"Feature importance plot saved: {plot_path}")

                # Also save as CSV
                csv_path = os.path.join(
                    self.model_training_config.feature_importance_dir,
                    f'{model_name.replace(" ", "_")}_importance.csv'
                )
                feat_importance.to_csv(csv_path, index=False)
                logging.info(f"Feature importance CSV saved: {csv_path}")

        except Exception as e:
            logging.error(f"Error saving feature importance for {model_name}: {str(e)}")

    def initiate_model_training(self, train_arr, test_arr):
        try:
            logging.info("Splitting Dependent and Independet variable form train and test data")
            x_train, y_train, x_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            ## Automation model training process
            models = {
                "Random Forest": RandomForestClassifier(),
                "Decision Tree": DecisionTreeClassifier(),
                "Gradient Classification": GradientBoostingClassifier()
            }

            model_report: dict = evaluate_mdoel(x_train, x_test, y_train, y_test, models)
            print("\n=======================================================================")
            logging.info(f"Model Reports: {model_report}")

            ## To get the best model score from the dictionary
            best_model_score = max(sorted(model_report.values()))

            # Find all models with the best score
            best_models = [name for name, score in model_report.items() if score == best_model_score]

            logging.info(f"Models with best score ({best_model_score}): {best_models}")

            # If multiple models have the same score, use feature importance to decide
            if len(best_models) > 1:
                logging.info("Multiple models have the same score. Using feature importance to select the best model.")
                print("\n🔍 Multiple models have the same score. Analyzing feature importance...")

                feature_scores = {}
                for model_name in best_models:
                    model = models[model_name]
                    # Fit the model first to get feature importances
                    model.fit(x_train, y_train)

                    # Calculate feature importance score
                    sig_features, entropy = self.calculate_feature_importance_score(model, model_name)
                    feature_scores[model_name] = (sig_features, entropy)
                    logging.info(f"Feature scores: {feature_scores}")

                    # Save feature importance plot
                    self.save_feature_importance_plot(model, model_name)

                    print(f"  {model_name}: Uses {sig_features} significant features, Entropy: {entropy:.4f}")

                # Select model with most significant features, then by entropy
                best_model_name = max(feature_scores.keys(),
                                     key=lambda x: (feature_scores[x][0], feature_scores[x][1]))

                logging.info(f"Selected {best_model_name} based on feature importance (uses most features)")
                print(f"\n✅ Selected {best_model_name} - uses the most features for robust predictions")
            else:
                best_model_name = best_models[0]
                logging.info(f"Single best model: {best_model_name}")

            best_model = models[best_model_name]

            # Make sure the final model is fitted
            best_model.fit(x_train, y_train)

            print(f"\n🎯 Best model: {best_model_name}, Score: {best_model_score}")
            print("\n=========================================================================")
            logging.info(f"Final best model: {best_model_name}, Score: {best_model_score}")

            save_obj(
                file_path=self.model_training_config.train_model_path,
                obj=best_model
            )

            logging.info(f"Model saved at: {self.model_training_config.train_model_path}")

            return best_model_score

        except Exception as e:
            logging.info("The error is Rasised in Data Model Training Stage")
            raise CustomException(e, sys)