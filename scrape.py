from bs4 import BeautifulSoup
import pandas as pd
import requests
import csv

# Load the CSV file containing blog data
df = pd.read_csv('data/blogs.csv')

#print(df)

def clean_html(raw_html):
    soup = BeautifulSoup(raw_html, "html.parser")
    # full_text = soup.get_text(separator=" ", strip=True)
    blog_text = soup.select("#uf-item-blog-content")

    if (len(blog_text) > 0 and blog_text[0].get_text(strip=True) != ""):
        cleaned_text = blog_text[0].get_text(strip=True)
        return cleaned_text
    else:
        return ""

def send_request(url):
    headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Referer': 'https://www.monster.com'
        }
    response = requests.get(url, headers=headers)
    return response


def get_blog_data(blog):
    print(f"Sending request {i+1}: {blog}")
    response = send_request(blog)
    if response.status_code != 200:
        print(f"Failed to retrieve blog {i+1}. Status code: {response.status_code}")
        print(response.text)
    markup = response.text
    cleaned_text = clean_html(markup)
    print(f"Cleaned text for blog {i+1}: {cleaned_text[:100]}...")  # Print first 100 characters
    print()
    return cleaned_text


with open('output.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    
    # Uncomment this if the file is new and you need headers
    # writer.writerow(['question', 'answer'])
    for i, row in df.iterrows():
        blog = row['blog']
        text = get_blog_data(blog)
        writer.writerow([text])

