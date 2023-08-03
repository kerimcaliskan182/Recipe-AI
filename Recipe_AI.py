import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import requests
from bs4 import BeautifulSoup

# Step 1: Load and preprocess the data
df = pd.read_csv('recipes.csv')
df['ingredients'] = df['ingredients'].apply(eval)
df['ingredients'] = df['ingredients'].apply(lambda x: ', '.join(x))
df['ingredients'] = df['ingredients'].apply(str.lower)

# Step 2: Create the feature vectors using TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['ingredients'])

# Step 3: Train the K-means clustering model
num_clusters = 3
kmeans = KMeans(n_clusters=num_clusters, n_init=10)
kmeans.fit(X)

# Step 4: Get the cluster labels for each recipe
df['cluster_label'] = kmeans.labels_

# Step 7: Build a recipe search function
def search_recipes_by_ingredients(user_input, dataset, n=5):
    user_input = [ingredient.lower() for ingredient in user_input]
    tfidf_vectorizer = TfidfVectorizer()
    user_input_str = ' '.join(user_input)
    user_input_vector = tfidf_vectorizer.fit_transform([user_input_str])
    recipe_vectors = tfidf_vectorizer.transform(dataset['ingredients'].apply(' '.join))
    similarity_scores = (user_input_vector * recipe_vectors.T).A[0]
    sorted_indices = similarity_scores.argsort()[::-1]
    top_n_indices = sorted_indices[:n]
    top_n_recipes = dataset.iloc[top_n_indices]
    return top_n_recipes

# Step 9: Identify missing ingredients
def identify_missing_ingredients(user_input, recipe_ingredients):
    missing_ingredients = [ingredient for ingredient in recipe_ingredients if ingredient not in user_input]
    return missing_ingredients

# Step 10: Provide missing ingredients to the user
user_ingredients = ['pasta', 'tomato sauce', 'cheese', 'garlic']
matching_recipes = search_recipes_by_ingredients(user_ingredients, df)


# Step 11: Find matching recipes based on provided ingredients
def find_matching_recipes(user_input_ingredients, dataset):
    matching_recipes = []
    for idx, recipe in dataset.iterrows():
        missing_ingredients = identify_missing_ingredients(user_input_ingredients, recipe['ingredients'])
        if not missing_ingredients:
            matching_recipes.append(recipe)
    return matching_recipes


# Step 12: User input and recipe display
# Step 12: User input and recipe display
def main():
    while True:
        print("Enter the ingredients you have (separated by commas, e.g., pasta, tomato sauce, cheese, garlic):")
        user_input_ingredients = input().strip().lower().split(',')

        matching_recipes = find_matching_recipes(user_input_ingredients, df)
        if matching_recipes:
            print("Matching Recipes:")
            for idx, recipe in enumerate(matching_recipes):
                missing_ingredients = identify_missing_ingredients(user_input_ingredients, recipe['ingredients'])
                print(f"Recipe {idx + 1}: {recipe['title']}")
                if missing_ingredients:
                    print(f"! Missing {len(missing_ingredients)} ingredient(s) !")
                else:
                    print("! No missing ingredient !")
                print()
        else:
            print("No matching recipes found for the provided ingredients.")
            print()

        print("Enter the name of the recipe you want to see (or 'exit' to quit):")
        user_input_recipe = input().strip().lower()

        if user_input_recipe == 'exit':
            break

        selected_recipe = matching_recipes[0] if matching_recipes else None
        if selected_recipe and selected_recipe['title'].lower().find(user_input_recipe) != -1:
            print(f"Recipe: {selected_recipe['title']}")

            # Fetch the recipe details from the website link
            recipe_link = selected_recipe['link']
            recipe_details = fetch_recipe_details(recipe_link)
            print(f"Ingredients: {', '.join(recipe_details['ingredients'])}")
            print(f"Directions: {recipe_details['directions']}")
            print(f"Link: {recipe_link}")

            missing_ingredients = identify_missing_ingredients(user_input_ingredients, recipe_details['ingredients'])
            if missing_ingredients:
                print(f"Missing ingredients: {', '.join(missing_ingredients)}")
            else:
                print("You have all the ingredients!")

            print()
        else:
            print("Recipe not found. Please try again.")
            print()


def fetch_recipe_details(recipe_link):
    page = requests.get(recipe_link)
    soup = BeautifulSoup(page.content, 'html.parser')

    # Find the element containing the ingredients
    ingredients_element = soup.find('div', {'class': 'recipe-ingredients'})
    ingredients = [ingredient.get_text().strip() for ingredient in ingredients_element.find_all('li')]

    # Find the element containing the preparation/directions
    directions_element = soup.find('div', {'class': 'recipe-directions'})
    directions = [direction.get_text().strip() for direction in directions_element.find_all('li')]

    return {'ingredients': ingredients, 'directions': directions}


if __name__ == "__main__":
    main()
