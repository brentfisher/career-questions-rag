import csv
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

with open('blog_output.csv', newline='', encoding='utf-8') as csvfile, \
     open('blog_summary.csv', mode='w', newline='', encoding='utf-8') as outfile:

    reader = csv.reader(csvfile)
    writer = csv.writer(outfile)

    for idx, row in enumerate(reader):
        question = row[0]
        summ = summarizer(question, max_length=130, min_length=30, do_sample=False)
        summary_text = summ[0]['summary_text']
        writer.writerow([summary_text])  # wrap in list to write as one cell

