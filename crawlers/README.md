# Project Overview
These scripts are designed to crawl movie plots from various sources such as **Wikipedia**, **IMDb**, and **Letterboxd**. Each sript targets a specific platform, fetching plot by searching for movie titles and saving the results into a structured format.

### wikipedia_crawler.py
This script uses Google search and Wikipedia to find and scrape movie plots. Movies that cannot be found during the search are saved as **'no plots'**.

### imdb_crawler.py
This script is desinged to crawl movie plots from IMDb using a list of movie titles form the MovieLens 1M dataset. The script searches for each movie on IMDb, retrieves the corresponding IMDb ID, and then scrapes the plot for the movie.

Sometimes the crawler includes irrelavant IMDb movie ID during the run. It arises due to movies with similar titles being retrieved during the search. To avoid such situations one can use the `--exact` flag:
```python
python imdb_crawler.py --exact True
```

### letterboxd_crawler.py
This script aims to crwal movie plots from Letterboxd for movies that do not already have plot data in the dataset. The dataset being used already has some movies with plots, but this script focuses on those without any plot information. It performs a search for each movie on Letterboxd, extracts the relevant plot, and saves the results.