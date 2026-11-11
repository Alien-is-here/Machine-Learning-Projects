# 📚 Book Haven – Book Recommender System

**Book Haven** is a Python-based Book Recommender System that helps users discover books based on popularity and similarity to their reading preferences.

The project uses **Collaborative Filtering** and **Cosine Similarity** to recommend five books similar to a book selected by the user. It also provides a Flask-based web application where users can explore popular books and get personalized book recommendations.

## ✨ Features

- **Top 50 Popular Books:** Displays the 50 most popular books based on rating count and average rating.
- **Book Recommendations:** Recommends five books similar to the user's selected book.
- **Collaborative Filtering:** Identifies similar books using user-rating patterns.
- **Cosine Similarity:** Measures similarity between books using their rating vectors.
- **Book Details:** Displays book titles, authors, cover images, and similarity scores.
- **Search Validation:** Handles empty input and book titles that are not found.
- **Web Interface:** Provides separate Home and Recommender pages using Flask and Jinja2.
- **Responsive Design:** Uses custom HTML and CSS with a vintage library-inspired theme.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | Cosine similarity calculation |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data analysis |
| YData Profiling | Automated exploratory data analysis |
| Flask | Web application backend |
| HTML5 | Webpage structure |
| CSS3 | Webpage styling |
| Jinja2 | Dynamic HTML rendering |
| Pickle | Saving and loading trained recommendation data |

## 📊 Dataset

The project uses three datasets:

- **Books.csv:** Contains book information such as ISBN, title, author, publication year, publisher, and image URLs.
- **Ratings.csv:** Contains user IDs, ISBNs, and book ratings.
- **Users.csv:** Contains user IDs, locations, and age information.

These datasets are used for exploratory data analysis, preprocessing, popularity calculations, and collaborative filtering.

*Note: The original CSV datasets are required to rerun the complete training and preprocessing notebook.*

## ⚙️ Recommendation System Workflow

### 1. Data Loading and Exploration

The three CSV datasets are loaded using Pandas. Exploratory Data Analysis (EDA) is performed to examine:

- Dataset dimensions and data types.
- Missing values and duplicate records.
- Statistical summaries.
- Distributions of publication years, ratings, and user ages.

### 2. Popularity-Based Recommendation

The popularity-based system:

1. Merges the books and ratings datasets using ISBN.
2. Calculates the number of ratings for each book.
3. Calculates the average rating for each book.
4. Keeps books with at least 250 ratings.
5. Sorts books by average rating in descending order.
6. Selects the top 50 books.

### 3. Collaborative Filtering

The collaborative filtering system identifies books based on user-rating patterns.

The preprocessing steps are:

1. Select users who have rated more than 200 books.
2. Identify books that have received at least 50 ratings from the filtered users.
3. Create a pivot table with book titles as rows and user IDs as columns.
4. Fill missing ratings with zero.

### 4. Cosine Similarity

Scikit-learn's `cosine_similarity()` calculates similarity between the book-rating vectors.

```python
similarity_score = cosine_similarity(pt)
```

When a user selects a book, the system:

1. Finds the selected book in the pivot table.
2. Retrieves its similarity scores.
3. Sorts books by similarity in descending order.
4. Excludes the selected book itself.
5. Returns the five most similar books.

### 5. Saving Recommendation Data

The processed DataFrames and similarity matrix are saved using Pickle:

- `popular.pkl`
- `pt.pkl`
- `similarity_score.pkl`
- `books.pkl`

These files allow the Flask application to load precomputed data instead of repeating the training and preprocessing steps every time the website starts.

## 🌐 Web Application

The Flask application provides two main pages:

| Route | Description |
|---|---|
| `/` | Displays the top 50 popular books. |
| `/recommend` | Allows users to search for a book and receive five similar recommendations. |

### Application Flow

1. The user opens the Home page and explores popular books.
2. The user navigates to the Recommender page.
3. The user enters a book title.
4. Flask receives the submitted title through a POST request.
5. The recommendation system retrieves the most similar books.
6. Flask passes the results to the Jinja2 template.
7. The webpage displays the recommended books.

## 📁 Project Structure

```text
Book-Recommender-System/
│
├── app.py
│
├── popular.pkl
├── pt.pkl
├── similarity_score.pkl
├── books.pkl
│
├── templates/
│   ├── index.html
│   └── recommender.html
│
├── Book_Recomender_System.ipynb
│
└── README.md
```

**File descriptions:**

- `app.py`: Flask backend containing the application routes and recommendation logic.
- `popular.pkl`: Stores the top 50 popular books.
- `pt.pkl`: Stores the book-user pivot table.
- `similarity_score.pkl`: Stores the cosine similarity matrix.
- `books.pkl`: Stores the original books DataFrame.
- `templates/index.html`: Home page displaying popular books.
- `templates/recommender.html`: Book search and recommendation page.
- `Book_Recomender_System.ipynb`: Notebook containing EDA, preprocessing, recommendation model development, and Pickle file generation.

## 🚀 Installation and Setup

### 1. Clone the Repository

```bash
git clone <your-repository-url>
```

### 2. Navigate to the Project Directory

```bash
cd Book-Recommender-System
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Virtual Environment

**Windows:**

```bash
.venv\Scripts\activate
```

**Linux / macOS:**

```bash
source .venv/bin/activate
```

### 5. Install Dependencies

For running the Flask application:

```bash
pip install flask pandas numpy scikit-learn
```

For running the complete Jupyter Notebook and EDA:

```bash
pip install matplotlib seaborn ydata-profiling notebook
```

### 6. Run the Flask Application

```bash
python app.py
```

### 7. Open the Website

Visit the following URL in your browser:

```text
http://127.0.0.1:5000/
```

The Home page displays popular books. Navigate to the Recommender page to search for a book and discover similar recommendations.

## 📌 Important Notes

- The four Pickle files must be present in the project root directory before running `app.py`.
- The CSV datasets are needed if you want to execute the complete preprocessing and model development notebook again.
- The recommender uses exact book-title matching, ignoring leading/trailing spaces and differences in letter case.
- The system returns up to five recommendations based on the available similarity scores.
- Flask debug mode is enabled for local development and should be disabled before production deployment.

## 🎯 Project Objective

The objective of this project is to understand and implement a practical recommendation system using Python, Pandas, and Scikit-learn, and deploy it through a Flask web application.

**Book Haven – Discover your next favorite book!**