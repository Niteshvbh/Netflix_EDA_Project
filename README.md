# 📊 Netflix EDA Project

An in-depth **Exploratory Data Analysis (EDA)** of the Netflix dataset to uncover trends in content production, genres, ratings, and more.  
This project includes **data cleaning, visualization, and insights** that can help understand Netflix’s content strategy over the years.

---

## 📂 Dataset
- **Source:** [Netflix Titles on Kaggle](https://www.kaggle.com/datasets/shivamb/netflix-shows)
- **File Used:** `netflix_titles.csv`
- **Description:** Contains details of movies and TV shows available on Netflix, including title, director, cast, country, release year, rating, duration, and listed genres.
- **Credits:** Dataset created and shared by **Shivam Bansal** on Kaggle.

---

## 🎯 Project Goals
1. Clean and preprocess raw Netflix dataset.
2. Perform **Exploratory Data Analysis (EDA)** to find patterns.
3. Visualize insights on:
   - Content distribution over years
   - Top genres & countries
   - Ratings and maturity levels
   - Keyword analysis via **WordCloud**
4. Prepare the project for **portfolio showcasing** on GitHub.

---

## 🧹 Data Cleaning & Preprocessing
- Removed **duplicate** records.
- Handled **missing values** for important fields like `country` and `rating`.
- Standardized **ratings** into a `rating_cleaned` column.
- Fixed **inconsistent date formats**.
- Saved cleaned dataset as `netflix_cleaned.csv`.

---

## 📈 Exploratory Data Analysis (EDA)
### Key Visualizations:
- 📊 **Movies vs TV Shows** distribution.
- 📅 **Content trend over the years**.
- 🌍 **Top countries producing content**.
- 🎭 **Most popular genres**.
- 🔞 **Distribution of ratings**.
- ☁ **WordCloud** for titles.

---

## ⚙ Installation & Usage
```bash
# Clone the repository
git clone https://github.com/your-username/Netflix_EDA_Project.git

# Navigate into the folder
cd Netflix_EDA_Project

# Install dependencies
pip install -r requirements.txt

# Run the notebook or script
jupyter notebook notebook.ipynb
# OR
python netflix_eda.py
```

📊 Results & Insights
Netflix's content production increased significantly after 2015.

The United States dominates production, followed by India and the UK.

Dramas and Comedies are the most popular genres.

TV-MA rating (Mature Audience) is most common.

🛠 Technologies Used
Python (pandas, numpy, matplotlib, seaborn)

Jupyter Notebook / Spyder

WordCloud

Git & GitHub for version control

🚀 Future Work
Add interactive dashboards (Plotly, Dash, or Tableau).

Expand dataset analysis with NLP on descriptions.

Compare Netflix with other streaming platforms.

👨‍💻 Author
Nitesh
- # 📧 Contact: niteshvbhoye@gmail.com
- # 🔗 LinkedIn: https://www.linkedin.com/in/nitesh-bhoye-567a2414a/
