# Titanic_Survival_Rate
This project focuses on predicting the survival of passengers on the Titanic using various machine learning techniques. By analyzing a dataset containing passenger information such as age, gender, passenger class, and more, we aim to build a robust model that can accurately determine the likelihood of survival.

## ✌️Problem Statement

The sinking of the RMS Titanic in 1912 is one of the most infamous shipwrecks in history. A significant number of passengers and crew lost their lives. This project seeks to answer the question: "What factors made some people more likely to survive than others?" Using the provided dataset, we will develop a predictive model to identify key features influencing survival rates.

## 💯Dataset

The dataset used in this project is the classic "Titanic - Machine Learning from Disaster" dataset, commonly found on Kaggle. It contains various features for each passenger, including:

* `PassengerId`: A unique identifier for each passenger.
* `Survived`: Survival status (0 = No, 1 = Yes). This is our target variable.
* `Pclass`: Passenger class (1 = 1st, 2 = 2nd, 3 = 3rd).
* `Name`: Name of the passenger.
* `Sex`: Gender of the passenger.
* `Age`: Age of the passenger.
* `SibSp`: Number of siblings/spouses aboard the Titanic.
* `Parch`: Number of parents/children aboard the Titanic.
* `Ticket`: Ticket number.
* `Fare`: Passenger fare.
* `Cabin`: Cabin number.
* `Embarked`: Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## 😎Methodology

The project follows a standard machine learning pipeline:

1.  **Exploratory Data Analysis (EDA):** Initial analysis to understand the dataset's structure, identify missing values, and visualize distributions and relationships between features and the target variable.
2.  **Data Preprocessing:**
    * Handling missing values (e.g., imputing 'Age' with the median, 'Embarked' with the mode).
    * Feature Engineering: Extracting 'Title' from passenger names, creating a 'FamilySize' feature.
    * Encoding categorical variables ('Sex', 'Embarked', 'Title') into numerical representations using `LabelEncoder`.
    * Dropping irrelevant columns (`PassengerId`, `Name`, `Ticket`, `Cabin`).
3.  **Model Selection & Training:**
    * The dataset is split into training and testing sets (80% training, 20% testing).
    * A `RandomForestClassifier` is chosen as the primary model due to its robustness and good performance on various datasets.
    * `GridSearchCV` is employed to fine-tune the model's hyperparameters (`n_estimators`, `max_depth`) for optimal performance.
4.  **Evaluation:**
    * The model's performance is evaluated using standard classification metrics:
        * Accuracy Score
        * Confusion Matrix
        * Classification Report (Precision, Recall, F1-score)
5.  **Feature Importance Analysis:**
    * The importance of each feature in the trained Random Forest model is visualized to understand which factors contribute most to survival prediction.

## 👍Results

The model achieves a good accuracy in predicting survival. Key insights derived from feature importance analysis indicate that `Sex`, `Pclass`, `Age`, `Fare`, `Title`, and `FamilySize` are among the most significant factors influencing survival.

**Example Output (from running the script):**






## 📈How to Run the Code

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/Titanic-Survival-Prediction.git](https://github.com/YourUsername/Titanic-Survival-Prediction.git)
    cd Titanic-Survival-Prediction
    ```
    (Replace `YourUsername` with your actual GitHub username and `Titanic-Survival-Prediction` with your repository name.)

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Place the dataset:**
    Ensure you have the `Titanic-Dataset.csv` file in the root directory of the cloned repository. You can download it from [Kaggle](https://www.kaggle.com/c/titanic/data).

5.  **Run the script:**
    ```bash
    python titanic_survival_prediction.py
    ```

    This will execute the data processing, model training, evaluation, and generate plots saved in the project directory.

## Plots Generated

The script will save the following plots in your project directory:

* `survival_count.png`
* `survival_by_pclass.png`
* `survival_by_gender.png`
* `feature_importance.png`

## Contributing

Feel free to fork this repository, make improvements, and submit pull requests. Any suggestions or enhancements are welcome!

## License

This project is open-sourced under the MIT License.

