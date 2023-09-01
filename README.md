# Independent Study: Introduction to Deep Learning

"Independent Study: Introduction to Deep Learning" is an independent study class conducted by Krystal Ly and advised by Dr. Anthony Bonifonte. The main focus of the class is to explore Deep Learning through readings, online courses, and a central project. Throughout the course, we have a chance to apply data analysis skills including, but not limited to, web scraping, data wrangling, data visualization, data modeling, and so on. In addition, we explore the basic of neural network, back propagation, different activation functions, and so on. This report serves as a comprehensive summary of what have been done in the class. 

## Data
#### 1. Book Genome Dataset
The project uses the Book Genome Dataset. This can be accessed [here](https://grouplens.org/datasets/book-genome/).

#### 2. Genre Dataset
Since the project aims at completing a book genre classification task, it is necessary to get data of the genres. The data is scraped from the Goodreads website using the book's Goodreads link in the `metadata.json`.

## Repository Overview
|── README.md
|── Report.ipynb: Python notebook of the report
|── Report.html: HTML version of the report
├── data: include all of the data files and result files
├── images: include all images used in the report
|── SetUp.ipynb: Python notebook to set up the project
|── WebScraping.ipynb: Python notebook to scrape data from Goodreads
|── DataCleaning.ipynb: Python notebook to clean data
|── EDA.ipynb: Python notebook to conduct exploratory data analysis
|── Modeling.ipynb: Python notebook to conduct the modeling part
|── helper.py: Python file consisting of helper functions
|── requirements.txt: text file consisting of all required libraries for the project

## Getting Started
1. Run the following code to download all required libraries
    pip install -r requirements.txt
2. Run the files in the following order: SetUp.ipynb, WebScraping.ipynb, DataCleaning.ipynb, EDA.ipynb, Modeling.ipynb

## License
Distributed under the MIT License.