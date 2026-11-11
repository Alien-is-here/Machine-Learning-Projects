
from flask import Flask, render_template, request
import pickle
from pathlib import Path

app = Flask(__name__)

# Get the project directory
BASE_DIR = Path(__file__).resolve().parent

# Load trained recommendation data
with open(BASE_DIR / 'popular.pkl', 'rb') as file:
    popular_df = pickle.load(file)

with open(BASE_DIR / 'pt.pkl', 'rb') as file:
    pt = pickle.load(file)

with open(BASE_DIR / 'similarity_score.pkl', 'rb') as file:
    similarity_scores = pickle.load(file)

with open(BASE_DIR / 'books.pkl', 'rb') as file:
    books_df = pickle.load(file)


# ------------------------------------------------
# HOME PAGE: Display Top 50 Popular Books
# ------------------------------------------------

@app.route('/')
def index():

    # Select only the top 50 books
    top_books = popular_df.head(50)

    # Convert DataFrame into list of dictionaries
    books = top_books.to_dict(orient='records')

    return render_template(
        'index.html',
        books=books
    )


# ------------------------------------------------
# RECOMMENDER PAGE
# ------------------------------------------------

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():

    # Display the recommendation page
    if request.method == 'GET':
        return render_template('recommender.html')

    # Get book title from the form
    user_input = request.form.get('user_input', '').strip()

    if not user_input:
        return render_template(
            'recommender.html',
            error="Please enter a book title."
        )

    # Find matching book title
    matches = [
        title for title in pt.index
        if str(title).strip().casefold() == user_input.casefold()
    ]

    if not matches:
        return render_template(
            'recommender.html',
            error="Book not found. Please enter a valid book title."
        )

    # Get the matched book title
    book_name = matches[0]

    # Get the index of the selected book
    book_index = pt.index.get_loc(book_name)

    # Get similarity scores for the selected book
    if hasattr(similarity_scores, 'iloc'):
        scores = similarity_scores.iloc[book_index].values
    else:
        scores = similarity_scores[book_index]

    # Sort books by similarity, excluding the selected book itself
    similar_books = sorted(
        enumerate(scores),
        key=lambda x: x[1],
        reverse=True
    )

    similar_books = [
        (index, score)
        for index, score in similar_books
        if index != book_index
    ][:5]

    recommendations = []

    # Retrieve recommended book details
    for index, score in similar_books:

        recommended_title = pt.index[index]

        temp_df = books_df[
            books_df['Book-Title'] == recommended_title
        ]

        if not temp_df.empty:

            book = temp_df.iloc[0]

            recommendations.append({
                'Book-Title': book['Book-Title'],
                'Book-Author': book['Book-Author'],
                'Image-URL-M': book['Image-URL-M'],
                'similarity': round(float(score), 3)
            })

    return render_template(
        'recommender.html',
        recommendations=recommendations,
        searched_book=book_name
    )


# ------------------------------------------------
# RUN FLASK APPLICATION
# ------------------------------------------------

if __name__ == '__main__':
    app.run(debug=True)
