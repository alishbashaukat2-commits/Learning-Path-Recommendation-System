# Learning Path Recommendation System

## About the Project

This project is a **Machine Learning-based Learning Path Recommendation System** developed as part of my ML internship.

The main purpose of this project is to recommend courses to interns based on their previous learning interactions. The system uses **Collaborative Filtering and Matrix Factorization** to find learning patterns and generate personalized course recommendations.

## Objective

The objectives of this project are:

* Analyze intern learning interactions.
* Create an intern-course interaction matrix.
* Find hidden learning patterns using Matrix Factorization.
* Predict interest in courses that an intern has not interacted with.
* Generate personalized Top-5 course recommendations.

## Dataset

The project uses a simulated learning dataset containing:

* **200 interns**
* **30 courses**
* **2,808 learning interactions**

The interaction matrix contains interns as rows and courses as columns.

Interaction scores from **1 to 5** represent recorded learning interactions, while **0** represents no recorded interaction with that course.

## Method Used

The project follows this process:

```text
Learning Interaction Data
          ↓
Interaction Matrix
          ↓
Collaborative Filtering
          ↓
Matrix Factorization
          ↓
TruncatedSVD
          ↓
Predicted Scores
          ↓
Top-5 Course Recommendations
```

### Collaborative Filtering

Collaborative Filtering recommends courses by learning patterns from existing intern-course interactions.

### Matrix Factorization

Matrix Factorization reduces the interaction matrix into a smaller number of hidden patterns, also called latent factors.

### TruncatedSVD

I used **Truncated Singular Value Decomposition (TruncatedSVD)** with **10 components** to perform the Matrix Factorization.

## Model Evaluation

The model produced the following results:

| Metric | Result |
| ------ | -----: |
| MAE    | 1.0088 |
| MSE    | 1.5514 |
| RMSE   | 1.2455 |

## Personalization

The system generates different recommendations for different interns.

For example:

* **INT001** → Technical Interview Preparation
* **INT002** → SQL Fundamentals
* **INT003** → AWS Fundamentals
* **INT004** → Git and GitHub
* **INT005** → Feature Engineering

This shows that the system provides personalized recommendations instead of giving every intern the same courses.

## Final Results

The system generated **1,000 recommendations** in total:

**200 interns × 5 recommendations = 1,000 recommendations**

The final results are stored in:

`all_intern_recommendations.csv`

## Project Files

| File                                             | Description                        |
| ------------------------------------------------ | ---------------------------------- |
| `learning path recommendation.py`                | Main Python code                   |
| `course_metadata.csv`                            | Course information                 |
| `intern_learning_history.csv`                    | Intern learning interaction data   |
| `all_intern_recommendations.csv`                 | Final personalized recommendations |
| `Learning_Path_Recommendation_System_Report.pdf` | Project report                     |

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* TruncatedSVD
* Machine Learning
* Collaborative Filtering
* Matrix Factorization

## Limitations

* The dataset is simulated.
* The number of interactions is limited.
* A zero score means no recorded interaction, not necessarily dislike.
* New interns with no interaction history may be difficult to recommend courses for.
* Course content itself is not currently analyzed.

## Future Improvements

In the future, this system can be improved by:

* Using real learning-platform data.
* Adding course descriptions and content-based recommendations.
* Combining collaborative filtering with content-based filtering.
* Including intern career goals and preferences.
* Updating recommendations as new learning interactions are recorded.
* Using additional recommendation evaluation metrics.

## Conclusion

This project demonstrates how Machine Learning can be used to create a personalized learning recommendation system. By using Collaborative Filtering, Matrix Factorization, and TruncatedSVD, the system learns patterns from intern-course interactions and recommends courses based on predicted interest.

This project helped me understand the practical use of recommendation systems and Matrix Factorization in Machine Learning.
