import pandas as pd
import numpy as np

# Load the datasets
learning_data = pd.read_csv(
    r"E:\Desktop\remote internship\task 3\intern_learning_history.csv"
)

course_data = pd.read_csv(
    r"E:\Desktop\remote internship\task 3\course_metadata.csv"
)

# Check the learning history
print("Learning History:")
print(learning_data.head())

# Check the course information
print("\nCourse Metadata:")
print(course_data.head())

# Check dataset size
print("\nLearning data shape:", learning_data.shape)
print("Course data shape:", course_data.shape)

# Check missing values
print("\nMissing values in learning data:")
print(learning_data.isnull().sum())

print("\nMissing values in course data:")
print(course_data.isnull().sum())

# Check number of interns and courses
print("\nNumber of interns:", learning_data["Intern_ID"].nunique())
print("Number of courses:", course_data["Course_ID"].nunique())

# Check interaction scores
print("\nInteraction Score Counts:")
print(learning_data["Interaction_Score"].value_counts().sort_index())
# Create the intern-course interaction matrix
interaction_matrix = learning_data.pivot_table(
    index="Intern_ID",
    columns="Course_ID",
    values="Interaction_Score",
    fill_value=0
)

print("\nInteraction Matrix:")
print(interaction_matrix.head())

print("\nInteraction Matrix Shape:")
print(interaction_matrix.shape)
# Convert the interaction matrix into a NumPy array
matrix = interaction_matrix.values

print("\nMatrix for the recommendation model:")
print(matrix[:5])

print("\nMatrix shape:")
print(matrix.shape)

# Count known and unknown interactions
known_interactions = np.count_nonzero(matrix)
unknown_interactions = matrix.size - known_interactions

print("\nKnown interactions:", known_interactions)
print("Unknown interactions:", unknown_interactions)
from sklearn.decomposition import TruncatedSVD

# Create the Matrix Factorization model
model = TruncatedSVD(n_components=10, random_state=42)

# Learn hidden patterns from the interaction matrix
user_factors = model.fit_transform(matrix)

# Get the course factors
course_factors = model.components_

print("\nUser factors shape:")
print(user_factors.shape)

print("\nCourse factors shape:")
print(course_factors.shape)

print("\nExplained variance ratio:")
print(model.explained_variance_ratio_)

print("\nTotal explained variance:")
print(model.explained_variance_ratio_.sum())
# Reconstruct the interaction matrix
predicted_matrix = np.dot(user_factors, course_factors)

print("\nPredicted Matrix:")
print(predicted_matrix[:5])

print("\nPredicted Matrix Shape:")
print(predicted_matrix.shape)

# Check predictions for the first intern
print("\nPredicted scores for INT001:")
print(predicted_matrix[0])
# Function to recommend courses for an intern
def recommend_courses(intern_id, top_n=5):

    # Get the row number of the intern
    intern_index = interaction_matrix.index.get_loc(intern_id)

    # Get predicted scores for this intern
    predicted_scores = predicted_matrix[intern_index].copy()

    # Get courses the intern has already interacted with
    already_taken = interaction_matrix.loc[intern_id] > 0

    # Remove already taken courses
    predicted_scores[already_taken.values] = -np.inf

    # Get indexes of the highest predicted scores
    top_indexes = np.argsort(predicted_scores)[::-1][:top_n]

    # Get course IDs
    recommended_course_ids = interaction_matrix.columns[top_indexes]

    # Get predicted scores
    recommended_scores = predicted_scores[top_indexes]

    # Create recommendation table
    recommendations = course_data[
        course_data["Course_ID"].isin(recommended_course_ids)
    ].copy()

    # Keep the same order as the recommendation scores
    score_dict = dict(zip(recommended_course_ids, recommended_scores))
    recommendations["Predicted_Score"] = recommendations["Course_ID"].map(score_dict)

    recommendations = recommendations.sort_values(
        "Predicted_Score",
        ascending=False
    )

    return recommendations


# Test the recommendation system
recommendations = recommend_courses("INT001", top_n=5)

print("\nRecommended courses for INT001:")
print(recommendations.to_string(index=False))
# Generate recommendations for all interns

all_recommendations = []

for intern_id in interaction_matrix.index:

    recommendations = recommend_courses(intern_id, top_n=5)

    for _, row in recommendations.iterrows():

        all_recommendations.append({
            "Intern_ID": intern_id,
            "Course_ID": row["Course_ID"],
            "Course_Name": row["Course_Name"],
            "Category": row["Category"],
            "Difficulty": row["Difficulty"],
            "Duration_Hours": row["Duration_Hours"],
            "Predicted_Score": row["Predicted_Score"]
        })


all_recommendations = pd.DataFrame(all_recommendations)

print("\nAll Intern Recommendations:")
print(all_recommendations.head(20).to_string(index=False))

print("\nTotal recommendations generated:")
print(len(all_recommendations))
# Save recommendations to CSV

output_path = r"E:\Desktop\remote internship\task 3\all_intern_recommendations.csv"

all_recommendations.to_csv(output_path, index=False)

print("\nRecommendations saved to:")
print(output_path)
# Evaluate the recommendation model

known_values = matrix[matrix > 0]
predicted_values = predicted_matrix[matrix > 0]

from sklearn.metrics import mean_absolute_error, mean_squared_error

mae = mean_absolute_error(known_values, predicted_values)
mse = mean_squared_error(known_values, predicted_values)
rmse = np.sqrt(mse)

print("\nModel Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
import matplotlib.pyplot as plt

# Interaction score distribution

score_counts = learning_data["Interaction_Score"].value_counts().sort_index()

plt.figure(figsize=(8, 5))

plt.bar(score_counts.index, score_counts.values)

plt.xlabel("Interaction Score")
plt.ylabel("Number of Interactions")
plt.title("Distribution of Intern Learning Interaction Scores")

plt.xticks([1, 2, 3, 4, 5])

plt.show()
# Most recommended courses

top_courses = (
    all_recommendations["Course_Name"]
    .value_counts()
    .head(10)
    .sort_values()
)

plt.figure(figsize=(10, 6))

plt.barh(top_courses.index, top_courses.values)

plt.xlabel("Number of Recommendations")
plt.ylabel("Course")
plt.title("Top 10 Most Recommended Courses")

plt.tight_layout()

plt.show()
# Check recommendations for different interns

sample_interns = ["INT001", "INT002", "INT003", "INT004", "INT005"]

for intern_id in sample_interns:

    print("\n" + "=" * 60)
    print("Recommendations for", intern_id)
    print("=" * 60)

    recommendations = recommend_courses(intern_id, top_n=5)

    print(
        recommendations[
            ["Course_ID", "Course_Name", "Category", "Difficulty", "Predicted_Score"]
        ].to_string(index=False)
    )
    # Step 12: Verify final recommendation file

print("\n" + "=" * 60)
print("FINAL RECOMMENDATION FILE")
print("=" * 60)

print("Total recommendations:", len(all_recommendations))
print("Columns:", list(all_recommendations.columns))

print("\nFirst 10 recommendations:")
print(all_recommendations.head(10).to_string(index=False))
print("\n" + "=" * 60)
print("TASK 3 COMPLETED SUCCESSFULLY")
print("=" * 60)
print("Personalized recommendations generated for all interns.")
print("Final recommendation file: all_intern_recommendations.csv")